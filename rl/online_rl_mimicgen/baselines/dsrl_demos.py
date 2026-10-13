"""Expert demonstrations in the DSRL replay."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch


def expert_ratio_at(env_steps: int, *, start: float, end: float, steps: int,
                    fixed: float, adaptive: bool) -> float:
    """The campaign's schedule, verbatim (common.py TrainConfig.current_expert_ratio, the non-performance-gated branch)."""
    if not adaptive:
        return float(fixed)
    if env_steps >= steps:
        return float(end)
    return float(start - (start - end) * (env_steps / max(steps, 1)))


STORE_FORMAT = "v2-padtail"


def chunk_demo_rows(episodes, success, actions, action_steps: int, step_penalty: float):
    """Chunk transitions for every demo that succeeds, cut at its first success frame."""
    rows_prev, rows_cur, rows_next, chunks, rewards, dones = [], [], [], [], [], []
    bounds = np.flatnonzero(np.r_[True, episodes[1:] != episodes[:-1], True])
    for start, stop in zip(bounds[:-1], bounds[1:]):
        hit = np.flatnonzero(success[start:stop])
        if hit.size == 0:
            continue
        terminal = start + int(hit[0])
        for i in range(start, terminal + 1, action_steps):
            end = min(i + action_steps, terminal + 1)
            chunk = actions[i:end]
            if len(chunk) < action_steps:
                chunk = np.concatenate((chunk, np.repeat(chunk[-1:], action_steps - len(chunk), axis=0)))
            done = end > terminal
            rows_prev.append(max(i - action_steps, int(start)))
            rows_cur.append(i)
            rows_next.append(min(end, terminal))
            chunks.append(chunk.reshape(-1))
            rewards.append(1.0 if done else step_penalty * (end - i))
            dones.append(1.0 if done else 0.0)
    return (np.asarray(rows_prev), np.asarray(rows_cur), np.asarray(rows_next),
            np.asarray(chunks, dtype=np.float32).reshape(len(chunks), -1),
            np.asarray(rewards, dtype=np.float32), np.asarray(dones, dtype=np.float32))


class DemoStore:
    """Chunk transitions from the task's demo parquet, in the DSRL replay's own layout."""

    def __init__(self, obs, next_obs, action, reward, done, idx_prev, idx_cur, idx_next, cache):
        self.obs, self.next_obs, self.action = obs, next_obs, action
        self.reward, self.done = reward, done
        self.idx = {"prev_images": idx_prev, "images": idx_cur, "next_images": idx_next}
        self.cache = cache

    def __len__(self) -> int:
        return len(self.obs)

    def frames(self, key: str, rows: np.ndarray) -> np.ndarray:
        """(B, V, H, W, 3) uint8 for the demo rows, read from the mmap on demand."""
        return np.ascontiguousarray(self.cache[self.idx[key][rows]])

    @classmethod
    def build(cls, *, dataset_root: Path, pixel_cache_path: Path, flow, scaler, action_steps: int,
              step_penalty: float, device, capacity: int = 0, cache_dir: Path | None = None,
              prior_id: str = "", verbose: bool = True) -> "DemoStore":
        import hashlib
        import pyarrow.dataset as pads


        key = hashlib.sha1('|'.join([str(Path(dataset_root).resolve()), prior_id, str(action_steps),
                                     str(getattr(flow, 'horizon', '?')), str(step_penalty),
                                     str(capacity), STORE_FORMAT]).encode()).hexdigest()[:16]
        cache_file = None if cache_dir is None else Path(cache_dir) / f"dsrl_demostore_{key}.npz"
        if cache_file is not None and cache_file.exists():
            z = np.load(cache_file, mmap_mode="r")
            store = cls(z["obs"], z["next_obs"], z["action"], z["reward"], z["done"],
                        z["idx_prev"], z["idx_cur"], z["idx_next"],
                        np.load(pixel_cache_path, mmap_mode="r"))
            if verbose:
                print(f"[dsrl-demos] reused encoded store {cache_file.name}: {len(store)} rows "
                      f"(no re-encoding)", flush=True)
            return store

        table = (pads.dataset(Path(dataset_root) / "data", format="parquet")
                 .to_table(columns=["episode_index", "frame_index", "observation.state",
                                    "action", "success"])
                 .sort_by([("episode_index", "ascending"), ("frame_index", "ascending")]))
        episodes = np.asarray(table["episode_index"].combine_chunks())
        states = np.asarray(table["observation.state"].combine_chunks().to_pylist(), dtype=np.float32)
        actions = np.asarray(table["action"].combine_chunks().to_pylist(), dtype=np.float32)
        success = np.asarray(table["success"].combine_chunks(), dtype=bool)

        cache = np.load(pixel_cache_path, mmap_mode="r")
        if len(cache) != len(states):
            raise RuntimeError(
                f"pixel cache has {len(cache)} frames but the parquet has {len(states)}; the cache is "
                "indexed by global frame number, so a mismatch pairs every demo state with the wrong "
                "image (expert.py guards the same invariant).")


        rows_prev, rows_cur, rows_next, chunks, rewards, dones = chunk_demo_rows(
            episodes, success, actions, action_steps, step_penalty)
        chunks = list(chunks); rewards = list(rewards); dones = list(dones)
        if capacity and len(rows_cur) > capacity:
            rows_cur = rows_cur[:capacity]
            rows_next = rows_next[:capacity]; rows_prev = rows_prev[:capacity]
            chunks = chunks[:capacity]; rewards = rewards[:capacity]; dones = dones[:capacity]
        n = len(rows_cur)
        if n == 0:
            raise RuntimeError(f"no successful demo chunks found under {dataset_root}")


        from ..pixel import CAMERAS


        def observe(rows: np.ndarray, label: str) -> np.ndarray:
            out = np.empty((len(rows), 0), dtype=np.float32)
            acc = []
            for j, r in enumerate(rows):
                pix = np.ascontiguousarray(cache[r])
                cond = flow.cond({"pixels": {c: pix[i][None] for i, c in enumerate(CAMERAS)},
                                  "agent_pos": states[r][None]})
                st = np.asarray(scaler(states[r][None]), dtype=np.float32)
                acc.append(np.concatenate([st, cond.float().cpu().numpy()], axis=1))
                if verbose and (j + 1) % 2000 == 0:
                    print(f"[dsrl-demos]   {label}: {j + 1}/{len(rows)} encoded", flush=True)
            return np.concatenate(acc, axis=0).astype(np.float32) if acc else out

        if verbose:
            print(f"[dsrl-demos] {n} successful demo chunks from {Path(dataset_root).name}; "
                  f"encoding prior conditioning ...", flush=True)
        obs = observe(rows_cur, "obs")
        next_obs = observe(rows_next, "next_obs")
        pad = flow.pad(torch.as_tensor(np.stack(chunks), device=device)
                       .view(n, action_steps, -1)).cpu().numpy().astype(np.float32)
        store = cls(obs, next_obs, pad,
                    np.asarray(rewards, dtype=np.float32).reshape(-1, 1),
                    np.asarray(dones, dtype=np.float32).reshape(-1, 1),
                    rows_prev, rows_cur, rows_next, cache)
        if verbose:
            gb = (obs.nbytes + next_obs.nbytes + pad.nbytes) / 2**30
            print(f"[dsrl-demos] store ready: {n} rows, obs dim {obs.shape[1]}, action dim "
                  f"{pad.shape[1]}, {gb:.2f} GiB resident (frames stay on the mmap)", flush=True)
        if cache_file is not None:
            cache_file.parent.mkdir(parents=True, exist_ok=True)


            tmp = cache_file.with_name(cache_file.stem + ".part.npz")
            np.savez(tmp, obs=obs, next_obs=next_obs, action=pad,
                     reward=store.reward, done=store.done,
                     idx_prev=rows_prev, idx_cur=rows_cur, idx_next=rows_next)
            tmp.replace(cache_file)
            if verbose:
                print(f"[dsrl-demos] cached to {cache_file} -- later runs on this task reuse it",
                      flush=True)
        return store
