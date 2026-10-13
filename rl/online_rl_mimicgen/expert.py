from __future__ import annotations
import os

from pathlib import Path

import numpy as np
import pyarrow.dataset as pads

from .common import ReplayBuffer, StateScaler, fill_mc_return


def _matrix(column) -> np.ndarray:
    return np.asarray(column.combine_chunks().to_pylist(), dtype=np.float32)


def load_successful_expert_buffer(
    dataset_root: Path,
    scaler: StateScaler,
    action_steps: int,
    gamma: float,
    step_penalty: float = -0.01,
    capacity: int = 50_000,
    samples: int = 1,
    next_samples: int | None = None,
    pixel_cache: np.ndarray | None = None,
    dino_encode=None,
    seg_encode=None,
    next_feat=False,
    tok_encode=None,
    store_images=None,
    boot_image: bool = False,
    continuation: bool = False,
) -> ReplayBuffer:
    """Build chunk transitions, truncating every demo at its first successful chunk."""
    dataset = pads.dataset(dataset_root / "data", format="parquet")
    table = dataset.to_table(
        columns=["episode_index", "frame_index", "observation.state", "action", "success"]
    ).sort_by([("episode_index", "ascending"), ("frame_index", "ascending")])
    episodes = np.asarray(table["episode_index"].combine_chunks())
    states = _matrix(table["observation.state"])
    actions = _matrix(table["action"])
    success = np.asarray(table["success"].combine_chunks(), dtype=bool)
    state_dim = states.shape[1]
    chunk_dim = action_steps * actions.shape[1]


    if pixel_cache is not None and len(pixel_cache) != len(states):
        raise RuntimeError(
            f"pixel cache has {len(pixel_cache)} frames but the parquet has {len(states)}; the "
            "cache is indexed by global frame number, so a mismatch would pair every demo state "
            "with the wrong image.")
    feat_shape = None
    if seg_encode is not None and pixel_cache is None:
        raise RuntimeError("seg_encode needs the demo pixel cache")
    if seg_encode is not None:


        feat_shape = tuple(np.asarray(
            seg_encode(pixel_cache[:1], states[:1, :9])).shape[1:])
    if dino_encode is not None:
        if pixel_cache is None:
            raise RuntimeError(
                "dino_encode needs the demo pixel cache: prev_feat is encoded from cache[index - "
                "action_steps], and there is no other source of demonstration frames here.")


        from .dino_dvd import FRAME_FEAT_SHAPE

        feat_shape = FRAME_FEAT_SHAPE
    replay = ReplayBuffer(
        capacity=capacity, state_dim=state_dim, chunk_dim=chunk_dim, samples=samples,
        next_samples=next_samples,


        image_shape=(None if pixel_cache is None
                     or (store_images is False) else tuple(pixel_cache.shape[1:])),
        feat_shape=feat_shape,
        seg_feat=seg_encode is not None,


        boot_image=boot_image,


        continuation=continuation,


        tok_shape=(None if tok_encode is None
                   else tuple(np.asarray(tok_encode(np.asarray(pixel_cache[:1]))).shape[1:])),
    )

    boundaries = np.flatnonzero(np.r_[True, episodes[1:] != episodes[:-1], True])
    candidates: list[dict] = []
    for start, stop in zip(boundaries[:-1], boundaries[1:], strict=True):
        first_success = np.flatnonzero(success[start:stop])
        if first_success.size == 0:
            continue
        terminal = start + int(first_success[0])
        episode: list[dict] = []
        for index in range(start, terminal + 1, action_steps):
            end = min(index + action_steps, terminal + 1)
            chunk = actions[index:end]
            if len(chunk) < action_steps:
                chunk = np.concatenate((chunk, np.repeat(chunk[-1:], action_steps - len(chunk), axis=0)))
            next_index = min(end, terminal)
            is_done = end > terminal
            next_start = min(index + action_steps, terminal)
            next_end = min(next_start + action_steps, terminal + 1)
            next_chunk = actions[next_start:next_end]
            if len(next_chunk) == 0:
                next_chunk = chunk[-1:]
            if len(next_chunk) < action_steps:
                next_chunk = np.concatenate(
                    (next_chunk, np.repeat(next_chunk[-1:], action_steps - len(next_chunk), axis=0))
                )
            steps = end - index
            episode.append(
                {
                    "state": scaler(states[index]),
                    "base": chunk.reshape(-1),
                    "action": chunk.reshape(-1),
                    "next_state": scaler(states[next_index]),
                    "next_base": next_chunk.reshape(-1),
                    "reward": 1.0 if is_done else step_penalty * steps,
                    "done": float(is_done),
                    "discount": gamma**steps,
                    "data_source": 1.0,


                    "base_samples": np.tile(chunk.reshape(-1), samples),
                    "next_base_samples": np.tile(next_chunk.reshape(-1), samples if next_samples is None else next_samples),
                    **({} if pixel_cache is None or store_images is False else
                       {"image": np.asarray(pixel_cache[index]),
                        "next_image": np.asarray(pixel_cache[next_index])}),


                    **({} if (dino_encode is None and seg_encode is None) else
                       {"prev_index": max(index - action_steps, int(start)),
                        "cur_index": int(index),

                        "next_index": int(next_index)}),
                }
            )
        candidates.extend(fill_mc_return(episode))

    if len(candidates) > capacity:
        indices = np.random.default_rng(0).choice(len(candidates), capacity, replace=False)
        candidates = [candidates[int(index)] for index in indices]
    if seg_encode is not None:


        import hashlib
        from .seghead import CKPT as _SEG_CKPT, PIXEL_IMG as _SEG_IMG, SEG_FEAT_SHAPE as _SFS
        _stamp = "|".join(
            str(fp.stat().st_mtime_ns)
            for fp in sorted(_SEG_CKPT.rglob("*.safetensors")) + [_SEG_CKPT / "wm_sup224/norm.json"])
        _key = hashlib.md5(
            f"{dataset_root}|{capacity}|{action_steps}|{len(candidates)}|"
            f"{tuple(pixel_cache.shape[1:])}|{_SEG_IMG}|{_SEG_CKPT}|{_stamp}".encode()
        ).hexdigest()[:12]


        _key = f"{_key}_f{'x'.join(str(d) for d in feat_shape)}{'_nf' if next_feat else ''}"
        _cache = Path(os.environ.get("CACHE_ROOT", "cache")) / "seghead_expert_cache" / _key
        _prev_npy, _cur_npy = _cache / "prev.npy", _cache / "cur.npy"


        _next_npy = _cache / "next.npy"
        if _prev_npy.exists() and _cur_npy.exists() and (_next_npy.exists() or not next_feat):


            _pf = np.load(_prev_npy, mmap_mode="r")
            _cf = np.load(_cur_npy, mmap_mode="r")
            _nf = np.load(_next_npy, mmap_mode="r") if next_feat else None
            _want = (len(candidates), *_SFS)
            if (_pf.shape == _want and _cf.shape == _want
                    and (_nf is None or _nf.shape == _want)):
                for k_, (t_, a_, b_) in enumerate(zip(candidates, _pf, _cf, strict=True)):
                    t_.pop("prev_index", None); t_.pop("cur_index", None)
                    t_.pop("next_index", None)
                    t_["prev_feat"] = a_; t_["seg_feat"] = b_
                    if _nf is not None:
                        t_["next_feat"] = _nf[k_]
                print(f"[seghead-wm] expert maps mmapped from cache {_cache.name}", flush=True)
                if tok_encode is not None:


                    print("[latent-obs-ft] cache ignored: tokens are not cached", flush=True)
                else:
                    seg_encode = None
            else:
                print(f"[seghead-wm] ignoring cache {_cache.name}: shapes {_pf.shape}/{_cf.shape} "
                      f"!= {_want}; re-encoding", flush=True)
    if seg_encode is not None:


        block = 32
        prev = np.asarray([t.pop("prev_index") for t in candidates])
        cur = np.asarray([t.pop("cur_index") for t in candidates])


        nxt = (np.asarray([t.pop("next_index") for t in candidates])
               if (next_feat or tok_encode is not None) else None)
        if tok_encode is not None and nxt is None:
            raise RuntimeError("tok_encode needs next_index; pass next_feat=True as well")
        for offset in range(0, len(prev), block):
            sl = slice(offset, offset + block)
            pf = seg_encode(np.asarray(pixel_cache[prev[sl]]), states[prev[sl]][:, :9])
            cf = seg_encode(np.asarray(pixel_cache[cur[sl]]), states[cur[sl]][:, :9])
            for t, a_, b_ in zip(candidates[sl], pf, cf, strict=True):
                t["prev_feat"] = a_; t["seg_feat"] = b_
            if nxt is not None:
                for t, c_ in zip(candidates[sl],
                                 seg_encode(np.asarray(pixel_cache[nxt[sl]]),
                                            states[nxt[sl]][:, :9]), strict=True):
                    t["next_feat"] = c_
            if tok_encode is not None:


                tk = tok_encode(np.asarray(pixel_cache[cur[sl]]))
                tn = tok_encode(np.asarray(pixel_cache[nxt[sl]]))
                for t, a_, b_ in zip(candidates[sl], tk, tn, strict=True):
                    t["tok"] = a_; t["next_tok"] = b_
            if offset % (block * 40) == 0:
                print(f"[seghead-wm] expert maps {offset}/{len(prev)}", flush=True)


        try:
            _cache.mkdir(parents=True, exist_ok=True)
            _fields = [("prev", "prev_feat"), ("cur", "seg_feat")]
            if next_feat:
                _fields.append(("next", "next_feat"))
            for _name, _field in _fields:
                _tmp = _cache / f".{_name}.tmp.npy"
                _stack = np.stack([t_[_field] for t_ in candidates])
                np.save(_tmp, _stack)
                del _stack
                _tmp.rename(_cache / f"{_name}.npy")
            print(f"[seghead-wm] cached expert maps -> {_cache}", flush=True)
        except OSError as exc:
            print(f"[seghead-wm] WARNING: cache write failed ({exc}); continuing uncached",
                  flush=True)
    elif dino_encode is not None:

        block = 64
        prev = np.asarray([transition.pop("prev_index") for transition in candidates])
        for offset in range(0, len(prev), block):
            frames = np.asarray(pixel_cache[prev[offset:offset + block]])
            feats = dino_encode(frames)
            for transition, feat in zip(candidates[offset:offset + block], feats, strict=True):
                transition["prev_feat"] = feat
            if offset % (block * 20) == 0:
                print(f"[dino-wm] expert prev_feat {offset}/{len(prev)}", flush=True)
    for transition in candidates:
        replay.add(**transition)
    if replay.size == 0:
        raise RuntimeError(f"No successful expert transitions found in {dataset_root}")
    return replay
