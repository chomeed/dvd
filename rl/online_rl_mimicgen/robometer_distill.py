"""Continuous distillation of RoboMeter into V, on the ONLINE replay stream."""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import torch


def _linspace_idx(n: int, k: int) -> list[int]:
    """k indices spanning [0, n-1] inclusive of both ends -- the RBM clip convention."""
    if n <= k:
        return list(range(n))
    return [int(round(x)) for x in np.linspace(0, n - 1, k)]


class _Pool:
    """One STRATUM of the distillation set:"""
    __slots__ = ("X", "Y", "P", "capacity", "added")

    def __init__(self, capacity: int):
        self.X: list[np.ndarray] = []
        self.Y: list[float] = []


        self.P: list[bool] = []
        self.capacity = capacity
        self.added = 0

    def __len__(self) -> int:
        return len(self.X)

    def extend(self, xs, ys, pred: bool = False) -> int:
        for x, y in zip(xs, ys):
            self.P.append(bool(pred))
            if isinstance(x, tuple):
                self.X.append(x); self.Y.append(float(y))
                continue
            a = np.asarray(x, dtype=np.float32)

            self.X.append(a if a.ndim == 2 else a.reshape(-1))
            self.Y.append(float(y))

        self.added += len(xs)
        over = len(self.X) - self.capacity
        if over > 0:
            del self.X[:over]; del self.Y[:over]; del self.P[:over]
        return len(self.X)


class RoboMeterDistiller:
    """Accumulate an episode, score it with RoboMeter, regress V onto the scores."""

    def __init__(self, value_fn, optimizer, spool: str, device,
                 frames_per_clip: int = 64, every: int = 500, batch_size: int = 256,
                 capacity: int = 20000, task: str | None = None, warmup_episodes: int = 1,
                 loss_weight: float = 1.0, verbose: bool = True,
                 mix: "dict[str, float] | None" = None, target: str = "product",
                 aug_views: int = 0, min_pool: "str | int" = 0,
                 deferred_encode=None, extra_optimizer=None, after_step=None):
        self.value_fn, self.optimizer, self.device = value_fn, optimizer, device


        self.deferred_encode, self.extra_optimizer, self.after_step = deferred_encode, extra_optimizer, after_step
        self.frames_per_clip, self.every, self.batch_size = frames_per_clip, every, batch_size
        self.capacity, self.warmup_episodes = capacity, warmup_episodes
        self.loss_weight, self.verbose = loss_weight, verbose
        self.task = task
        self.spool = Path(spool); (self.spool/"req").mkdir(parents=True, exist_ok=True)
        (self.spool/"res").mkdir(parents=True, exist_ok=True)
        self.timeout_s = 120.0
        self._feats: list[np.ndarray] = []
        self._frames: list[np.ndarray] = []


        if target not in ("progress", "success", "product", "scaled", "gated"):
            raise ValueError(
                f"--value-distill-target must be progress|success|product|scaled|gated, got {target!r}")
        self.target = target


        self.aug_views = int(aug_views)


        self.fail_floor = 0.1


        self.natural = (mix == "natural")
        self.mix = ({"demo": 1.0, "succ": 1.0, "fail": 1.0} if self.natural
                    else dict(mix or {"demo": 0.25, "succ": 0.45, "fail": 0.30}))
        _online_cap = max(1, capacity // 2)
        self.pools: dict[str, _Pool] = {
            "demo": _Pool(capacity),
            "succ": _Pool(_online_cap),
            "fail": _Pool(_online_cap),
        }
        for k in self.mix:
            if k not in self.pools:
                raise ValueError(f"--value-distill-mix names unknown pool {k!r}; "
                                 f"valid: {sorted(self.pools)}")


        import uuid
        self._nonce = uuid.uuid4().hex[:8]
        self._consec_failures = 0
        self.disabled = False


        self._rng = np.random.default_rng(0xD15C11)


        _mp = str(min_pool).strip().lower()
        if _mp in ("", "0", "off", "none"):
            self.min_pool = 0
        elif _mp == "auto":
            self.min_pool = "auto"
        else:
            self.min_pool = int(_mp)
            if self.min_pool < 0:
                raise ValueError(f"--value-distill-min-pool must be auto or >= 0, got {min_pool!r}")
        if self.min_pool and self.natural:
            raise ValueError("--value-distill-min-pool gates the DECLARED mix; with mix=natural there "
                             "is no per-stratum share to wait for")
        if self.min_pool:
            if not any(self.mix.get(k, 0.0) > 0.0 for k in ("succ", "fail")):
                raise ValueError("--value-distill-min-pool needs a positive succ or fail share in "
                                 "--value-distill-mix; there is nothing online to wait for")


            _worst = self._gate_need(demo_present=True) | self._gate_need(demo_present=False)
            for k in _worst:
                _n = max(self._gate_need(demo_present=True).get(k, 0),
                         self._gate_need(demo_present=False).get(k, 0))
                if _n > self.pools[k].capacity:
                    raise ValueError(
                        f"--value-distill-min-pool: stratum {k!r} would need {_n} rows but holds at "
                        f"most {self.pools[k].capacity} (--value-distill-capacity // 2); the gate "
                        f"could never open")
        self.gate_open = not self.min_pool
        self.gate_opened_episode = 0 if self.gate_open else -1
        self.last_take: dict[str, int] = {}
        self.episodes = 0
        self.fits = 0
        self.pred_pairs = 0
        self.pred_failures = 0
        self.failures = 0
        self.last_loss = float("nan")
        self.last_score_s = float("nan")
        import collections
        self._sep = collections.defaultdict(list)
        self.ep_success_final = float("nan")
        self.ep_success_mean = float("nan")
        self.ep_progress_final = float("nan")
        self.ep_progress_max = float("nan")
        self.ep_progress_mean = float("nan")
        self._uid = 0

    @property
    def size(self) -> int:
        return sum(len(p) for p in self.pools.values())


    def _score(self, frames: np.ndarray) -> np.ndarray:
        """Hand the clip to the sidecar and wait for its scores."""
        import time
        self._uid += 1
        rid = f"{os.getpid()}_{self._nonce}_{self._uid:06d}"


        tmp = self.spool / "req" / f"{rid}.tmp"
        with open(tmp, "wb") as fh:
            np.save(fh, frames)
        tmp.rename(self.spool / "req" / f"{rid}.npy")
        out, err = self.spool / "res" / f"{rid}.npy", self.spool / "res" / f"{rid}.err"
        t0 = time.time()
        while time.time() - t0 < self.timeout_s:
            if out.exists():
                a = np.load(out); out.unlink(missing_ok=True)
                self.last_score_s = time.time() - t0
                return a
            if err.exists():
                msg = err.read_text()[-1200:]; err.unlink(missing_ok=True)

                raise RuntimeError(f"sidecar: {msg}")
            time.sleep(0.05)
        raise TimeoutError(f"sidecar did not answer within {self.timeout_s}s "
                           f"(is robometer_server running on {self.spool}?)")


    def on_step(self, frame: np.ndarray, proprio) -> None:
        """One env step."""
        if frame is None:
            return
        self._frames.append(np.asarray(frame, dtype=np.uint8))
        self._feats.append(None if proprio is None else
                           (proprio.detach().float().cpu().numpy().reshape(-1)
                            if torch.is_tensor(proprio) else np.asarray(proprio, np.float32).reshape(-1)))

    def add_episode(self, frames, proprios, encode_fn, success: bool = False, pred_fn=None) -> int:
        """Score one finished episode without copying it."""
        n = len(frames)
        if n < 4:
            return 0
        idx = _linspace_idx(n, self.frames_per_clip)
        clip = np.stack([np.asarray(frames[i], dtype=np.uint8) for i in idx])
        self._frames = []
        self._feats = []
        return self._finish(clip, idx, [proprios[i] for i in idx], encode_fn, success, pred_fn)

    def on_episode_end(self, encode_fn=None, success: bool = False) -> int:
        """Score the finished episode and push (V input, teacher progress) pairs."""
        n = len(self._frames)
        self._frames_cleared = n
        if n < 4:
            self._feats.clear(); self._frames.clear()
            return 0
        idx = _linspace_idx(n, self.frames_per_clip)
        clip = np.stack([self._frames[i] for i in idx])
        props = [self._feats[i] for i in idx]
        self._feats = []; self._frames = []
        return self._finish(clip, idx, props, encode_fn, success)


    MAX_CONSEC_FAILURES = 5

    def _reduce_heads(self, raw: np.ndarray):
        """(H, T) teacher output -> (target, progress, success_or_None)."""
        a = np.asarray(raw, dtype=np.float64)
        if a.ndim == 1:
            a = a[None, :]
        prog = a[0]
        succ = a[1] if a.shape[0] > 1 else None
        if self.target == "gated":
            return prog, prog, succ
        if self.target == "progress" or succ is None:
            if self.target != "progress" and succ is None:


                raise RuntimeError(
                    f"--value-distill-target {self.target} needs the success head, but the sidecar "
                    f"returned progress only. Restart robometer_server with the two-row response.")
            return prog, prog, succ
        if self.target == "success":
            return succ, prog, succ
        if self.target == "product":
            return prog * succ, prog, succ


        scale = float(np.clip(succ[-1], self.fail_floor, 1.0))
        return prog * scale, prog, succ

    def _finish(self, clip, idx, props, encode_fn, success: bool = False, pred_fn=None) -> int:
        if self.disabled:
            return 0
        try:
            tgt, prog, succ = self._reduce_heads(self._score(clip))
        except Exception as e:
            self.failures += 1
            self._consec_failures += 1
            if self._consec_failures >= self.MAX_CONSEC_FAILURES:
                self.disabled = True
                print(f"[rm-distill] DISABLED after {self._consec_failures} consecutive failures "
                      f"-- V keeps its current weights and training continues. "
                      f"rm_distill_failures stays visible in the metrics.", flush=True)
            print(f"[rm-distill] scoring failed ({type(e).__name__}: {e}); episode skipped",
                  flush=True)
            return 0
        if len(tgt) != len(idx):


            self.failures += 1
            self._consec_failures += 1
            print(f"[rm-distill] teacher returned {len(tgt)} scores for {len(idx)} frames; "
                  f"episode skipped (a partial reply cannot be aligned)", flush=True)
            return 0
        self._consec_failures = 0
        m = len(idx)


        try:


            vin = encode_fn(clip[:m], props[:m], idx[:m])
            if self.aug_views > 0:
                _v = [vin.detach().float().cpu().numpy() if torch.is_tensor(vin) else np.asarray(vin)]
                for _ in range(self.aug_views):
                    _a = encode_fn(clip[:m], props[:m], idx[:m], aug=True)
                    _v.append(_a.detach().float().cpu().numpy() if torch.is_tensor(_a) else np.asarray(_a))
                vin = np.stack(_v, axis=1)
        except Exception as e:
            self.failures += 1
            self._consec_failures += 1
            print(f"[rm-distill] encode_fn failed ({type(e).__name__}: {e}); episode skipped",
                  flush=True)
            return 0
        if isinstance(vin, tuple):
            if self.deferred_encode is None or self.aug_views > 0:
                raise RuntimeError("encode_fn returned raw inputs but deferred encoding is off (or aug_views > 0)")
            _fr, _pp = vin
            vin = [(np.ascontiguousarray(_fr[j], dtype=np.uint8), np.asarray(_pp[j], np.float32)) for j in range(m)]
        else:
            vin = vin.detach().float().cpu().numpy() if torch.is_tensor(vin) else np.asarray(vin)
        if self.target == "gated" and not success:
            _s = float(np.clip(succ[-1], self.fail_floor, 1.0)) if succ is not None else self.fail_floor
            tgt = prog * _s
        pool = "succ" if success else "fail"
        n_pool = self.pools[pool].extend([vin[j] for j in range(m)], [tgt[j] for j in range(m)])
        if pred_fn is not None:


            try:
                vp, pos = pred_fn(idx[:m])
                vp = vp.detach().float().cpu().numpy() if torch.is_tensor(vp) else np.asarray(vp)
                if len(pos):
                    self.pools[pool].extend([vp[k] for k in range(len(pos))], [tgt[p] for p in pos], pred=True)
                    self.pred_pairs += len(pos)
            except Exception as e:
                self.pred_failures += 1
                print(f"[rm-distill] pred_fn failed ({type(e).__name__}: {e}); actual pairs kept", flush=True)


        self.ep_success_final = float(succ[m - 1]) if succ is not None else float("nan")
        self.ep_success_mean = float(np.mean(succ[:m])) if succ is not None else float("nan")
        self._sep[pool].append(self.ep_success_final)
        self.episodes += 1


        self.ep_progress_final = float(prog[m - 1])
        self.ep_progress_max = float(np.max(prog[:m]))
        self.ep_progress_mean = float(np.mean(prog[:m]))
        if self.verbose:
            print(f"[rm-distill] episode {self.episodes} [{pool}]: +{m} pairs "
                  f"(progress {prog.min():.3f}..{prog.max():.3f}), pools "
                  + " ".join(f"{k}={len(v)}" for k, v in self.pools.items()), flush=True)
        return m


    def seed_offline(self, clips, proprios_per_clip, encode_fn) -> int:
        """Fill the `demo` pool from OFFLINE expert demonstrations, scored once and cached."""
        added = 0
        for clip, props in zip(clips, proprios_per_clip):
            n = len(clip)
            idx = _linspace_idx(n, self.frames_per_clip)
            sub = np.stack([np.asarray(clip[i], dtype=np.uint8) for i in idx])
            try:
                prog = self._score(sub)
            except Exception as e:
                print(f"[rm-distill] demo scoring failed ({type(e).__name__}: {e}); "
                      f"seeded {added} pairs before giving up", flush=True)
                break
            if len(prog) != len(idx):
                continue
            try:
                vin = encode_fn(sub, [props[i] for i in idx], idx)
            except Exception as e:
                print(f"[rm-distill] demo encode failed ({type(e).__name__}: {e})", flush=True)
                break
            vin = vin.detach().float().cpu().numpy() if torch.is_tensor(vin) else np.asarray(vin)
            added += len(idx)
            self.pools["demo"].extend([vin[j] for j in range(len(idx))], list(prog))
        print(f"[rm-distill] demo pool seeded: {len(self.pools['demo'])} pairs "
              f"from {len(clips)} expert episodes", flush=True)
        return added

    def _view(self, x: np.ndarray, clean: bool) -> np.ndarray:
        if isinstance(x, tuple):
            return x
        if x.ndim == 1:
            return x
        if clean or x.shape[0] == 1:
            return x[0]
        return x[1 + int(self._rng.integers(x.shape[0] - 1))]

    def _sample(self, n: int, clean: bool = False):
        """Draw a batch with the DECLARED composition, not the policy's incidental one."""
        avail = {k: p for k, p in self.pools.items() if len(p) and self.mix.get(k, 0.0) > 0.0}
        if not avail:
            return None, None, {}
        w = ({k: float(len(p)) for k, p in avail.items()} if self.natural
             else {k: self.mix[k] for k in avail})
        tot = sum(w.values())
        want = {k: int(round(n * w[k] / tot)) for k in avail}
        take = {k: min(want[k], len(avail[k])) for k in avail}
        short = n - sum(take.values())
        if short > 0:
            for k in sorted(avail, key=lambda k: len(avail[k]) - take[k], reverse=True):
                room = len(avail[k]) - take[k]
                if room <= 0:
                    continue
                add = min(room, short); take[k] += add; short -= add
                if short == 0:
                    break
        xs, ys = [], []
        for k, cnt in take.items():
            if cnt <= 0:
                continue
            pool = avail[k]
            sel = self._rng.choice(len(pool), cnt, replace=False)
            xs.extend(self._view(pool.X[i], clean) for i in sel); ys.extend(pool.Y[i] for i in sel)
        if not xs:
            return None, None, {}
        if isinstance(xs[0], tuple):
            return (np.stack([x[0] for x in xs]), np.stack([x[1] for x in xs])), np.asarray(ys, dtype=np.float32), take
        return np.stack(xs), np.asarray(ys, dtype=np.float32), take


    def _member_preds(self, x: torch.Tensor) -> torch.Tensor:
        """Per-ENSEMBLE-MEMBER predictions, (members, B, 1), falling back to V's own output."""
        nets = getattr(self.value_fn, "nets", None)
        if nets is None or getattr(self.value_fn, "_delta", None) is not None:

            return self.value_fn(x)
        return torch.stack([net(x) for net in nets], dim=0)

    def _gate_need(self, demo_present: "bool | None" = None) -> dict:
        """Rows each gated stratum must hold before the first fit (see MIN-POOL GATE)."""
        online = {k: self.mix.get(k, 0.0) for k in ("succ", "fail") if self.mix.get(k, 0.0) > 0.0}
        if not online or not self.min_pool:
            return {}
        if demo_present is None:
            demo_present = len(self.pools["demo"]) > 0
        tot = sum(online.values()) + (self.mix.get("demo", 0.0) if demo_present else 0.0)
        need = {k: int(round(self.batch_size * v / tot)) for k, v in online.items()}
        if self.min_pool != "auto":
            need = {k: max(n, int(self.min_pool)) for k, n in need.items()}
        return need

    def maybe_fit(self, updates: int) -> float:
        if (self.episodes < self.warmup_episodes or not self.size
                or self.every <= 0 or updates % self.every):
            return float("nan")
        if not self.gate_open:
            need = self._gate_need()
            if any(len(self.pools[k]) < n for k, n in need.items()):
                return float("nan")
            self.gate_open, self.gate_opened_episode = True, self.episodes
            if self.verbose:
                print(f"[rm-distill] min-pool gate OPEN after {self.episodes} episodes: "
                      + " ".join(f"{k}={len(self.pools[k])}>={n}" for k, n in need.items()),
                      flush=True)
        xb, yb, take = self._sample(min(self.batch_size, self.size))
        if xb is None:
            return float("nan")
        self.last_take = take
        if self.extra_optimizer is not None:
            self.extra_optimizer.zero_grad(set_to_none=True)
        self.optimizer.zero_grad(set_to_none=True)
        if isinstance(xb, tuple):
            x = self.deferred_encode(xb[0], xb[1], True)
        else:
            x = torch.as_tensor(xb, device=self.device, dtype=torch.float32)
        y = torch.as_tensor(yb, device=self.device).unsqueeze(-1)
        pred = self._member_preds(x)
        if pred.dim() == 3:
            y = y.unsqueeze(0).expand_as(pred)
        _lf = torch.nn.functional.mse_loss if getattr(self, "value_loss", "smooth_l1") == "mse" else torch.nn.functional.smooth_l1_loss
        loss = self.loss_weight * _lf(pred, y)
        loss.backward()
        self.optimizer.step()
        if self.extra_optimizer is not None:
            self.extra_optimizer.step()
            self.extra_optimizer.zero_grad(set_to_none=True)
        if self.after_step is not None:
            self.after_step()
        self.fits += 1
        self.last_loss = float(loss.detach())
        return self.last_loss

    @torch.no_grad()
    def probe(self, n: int = 512) -> dict:
        """Read on whether V TRACKS the teacher -- no gradient step."""
        if not self.size:
            return {}
        xb, yb, _ = self._sample(min(n, self.size), clean=True)
        if xb is None:
            return {}
        k = len(yb)
        x = (self.deferred_encode(xb[0], xb[1], False) if isinstance(xb, tuple)
             else torch.as_tensor(xb, device=self.device, dtype=torch.float32))
        y = yb.astype(np.float64)


        v = self.value_fn(x)
        if v.dim() == 3:
            v = v.mean(0)
        v = v.detach().float().cpu().numpy().reshape(-1)[:k]
        out = {"rm_distill_gap": float(np.mean(np.abs(v - y))),
               "rm_distill_v_mean": float(np.mean(v)), "rm_distill_v_std": float(np.std(v)),
               "rm_distill_probe_n": float(k)}


        if np.std(v) > 1e-8 and np.std(y) > 1e-8:
            out["rm_distill_corr"] = float(np.corrcoef(v, y)[0, 1])
        else:
            out["rm_distill_corr"] = 0.0


        for name, pool in self.pools.items():
            if not len(pool):
                continue
            kk = min(256, len(pool))
            sel = self._rng.choice(len(pool), kk, replace=False)
            _rows = [self._view(pool.X[i], True) for i in sel]
            if isinstance(_rows[0], tuple):
                xx = self.deferred_encode(np.stack([r[0] for r in _rows]), np.stack([r[1] for r in _rows]), False)
            else:
                xx = torch.as_tensor(np.stack(_rows), device=self.device, dtype=torch.float32)
            yy = np.asarray([pool.Y[i] for i in sel], dtype=np.float64)
            vv = self.value_fn(xx)
            if vv.dim() == 3:
                vv = vv.mean(0)
            vv = vv.detach().float().cpu().numpy().reshape(-1)[:kk]
            out[f"rm_distill_{name}_gap"] = float(np.mean(np.abs(vv - yy)))
            out[f"rm_distill_{name}_target_mean"] = float(np.mean(yy))
            out[f"rm_distill_{name}_corr"] = (
                float(np.corrcoef(vv, yy)[0, 1])
                if np.std(vv) > 1e-8 and np.std(yy) > 1e-8 else 0.0)


            is_pred = np.asarray([pool.P[i] if i < len(pool.P) else False for i in sel], dtype=bool)
            if (~is_pred).any():
                out[f"rm_distill_{name}_v_mean"] = float(np.mean(vv[~is_pred]))
                out[f"rm_distill_{name}_real_n"] = float((~is_pred).sum())
            if is_pred.any():
                out[f"rm_distill_{name}_pred_v_mean"] = float(np.mean(vv[is_pred]))
        if "rm_distill_succ_v_mean" in out and "rm_distill_fail_v_mean" in out:


            out["rm_distill_v_sep"] = out["rm_distill_succ_v_mean"] - out["rm_distill_fail_v_mean"]
        return out

    def _separation(self) -> float:
        """success-head reading on succeeded episodes minus failed ones, over a trailing window."""
        S, F = self._sep["succ"][-100:], self._sep["fail"][-100:]
        S = [x for x in S if x == x]; F = [x for x in F if x == x]
        if not S or not F:
            return float("nan")
        return float(np.mean(S) - np.mean(F))

    def stats(self, probe_n: int = 512) -> dict:
        """Scalars for the training log."""
        d = {"rm_distill_episodes": float(self.episodes),
             "rm_distill_fits": float(self.fits),
             "rm_distill_failures": float(self.failures),
             "rm_distill_pred_pairs": float(getattr(self, "pred_pairs", 0)),
             "rm_distill_pred_failures": float(getattr(self, "pred_failures", 0)),
             "rm_distill_buffer": float(self.size),
             "rm_distill_gate_open": float(self.gate_open),
             "rm_distill_gate_episode": float(self.gate_opened_episode),
             "rm_distill_last_loss": self.last_loss,
             "rm_distill_score_s": self.last_score_s,
             "rm_distill_ep_success_final": self.ep_success_final,
             "rm_distill_ep_success_mean": self.ep_success_mean,


             "rm_distill_head_separation": self._separation(),
             "rm_distill_ep_progress_final": self.ep_progress_final,
             "rm_distill_ep_progress_max": self.ep_progress_max,
             "rm_distill_ep_progress_mean": self.ep_progress_mean,
             **{f"rm_distill_n_{k}": float(len(v)) for k, v in self.pools.items()},


             **{f"rm_distill_batch_{k}": float(self.last_take.get(k, 0))
                for k in self.pools}}
        _allY = [y for pool in self.pools.values() for y in pool.Y]
        d["rm_distill_target_mean"] = float(np.mean(_allY)) if _allY else float("nan")
        d["rm_distill_target_std"] = float(np.std(_allY)) if _allY else float("nan")
        if probe_n:
            d.update(self.probe(probe_n))
        return d
