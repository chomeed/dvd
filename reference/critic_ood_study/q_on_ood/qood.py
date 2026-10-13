"""Q on nearby out-of-distribution actions, per decision of one episode (DSRL hammer_cleanup_d1, Q arm).

phase ref   : roll one episode with the checkpoint's own actor (single sampled noise per chunk, the eval protocol).
              At every decision t: an exact simulator snapshot, the executed action chunk a_t, and the critic's
              Q^A(s_t, a_t + delta) for every perturbation delta (all 10 heads). Every primitive-step camera frame is
              kept for the video.
phase gt    : for each (t, delta): restore the snapshot, execute a_t + delta, then follow the same actor to the end of
              the episode; N rollouts with different seeds. Ground truth Q_MC = mean of gamma8^k (k = index of the
              chunk in which the task succeeds, 0 = the perturbed chunk itself; timeout = 0), the quantity Q^A is
              trained to predict (gamma^8 per chunk, reward 1 on the success chunk, timeouts terminal).
              The first chunk is deterministic, so the end-effector displacement it causes is reported in mm.
Perturbation: a constant offset m added to the normalised position command of one axis for all 8 steps.
"""
from __future__ import annotations

import argparse
import collections
import copy
import hashlib
import json
import os
import pickle
import random
import sys
import time
import zlib
from pathlib import Path

import numpy as np
import torch

R = Path("/NHNHOME/WORKSPACE/26moe002_B/chomeed/workspace/ICLR/online_rl_mimicgen")
for p in (str(R), str(R / "0811")):
    if p not in sys.path:
        sys.path.insert(0, p)

from online_rl_mimicgen.baselines.dsrl_dino import DinoFrameEnv, DSRLSVGDino, ExecutedNoisePrior, PixelProprioSlice  # noqa: E402
from online_rl_mimicgen.baselines.dsrl_mimicgen import FlowNoisePrior, StateSlice  # noqa: E402
from online_rl_mimicgen.common import TASKS, FrozenPolicyPrior, StateScaler, find_latest_checkpoint  # noqa: E402
from stable_baselines3.common.save_util import load_from_zip_file  # noqa: E402
from stable_baselines3.common.vec_env import DummyVecEnv  # noqa: E402

GAMMA8 = 0.99 ** 8


# ------------------------------------------------------------------ model (verbatim eval_dsrl_ckpt construction)
def build(ckpt: Path, seed: int, with_dvd: bool = False):
    run = ckpt.resolve().parents[1]
    cfg = json.loads((run / "config.json").read_text())
    task_cfg = TASKS[cfg["task"]]
    state_dim = int(task_cfg["state_dim"])
    prior = FrozenPolicyPrior(cfg["task"], find_latest_checkpoint(Path(cfg["base_output"])), cfg["action_steps"],
                              cfg["inference_steps"])
    flow = FlowNoisePrior(prior, state_dim)
    if cfg.get("executed_noise_only", "off") != "off":
        flow = ExecutedNoisePrior(flow, cfg["executed_noise_only"])
    env = DinoFrameEnv(cfg["task"], flow, StateScaler(task_cfg["dataset_root"]), seed=seed, variant="na",
                       magnitude=cfg["action_magnitude"], step_penalty=cfg["step_penalty"],
                       dense_reward=cfg["env_dense_reward"], horizon=cfg["episode_horizon"])
    net = [cfg["layer_size"]] * cfg["num_layers"]
    extractor = dict(features_extractor_class=PixelProprioSlice,
                     features_extractor_kwargs=dict(out_dim=cfg["pixel_feature_dim"], shift_pad=cfg["pixel_shift_pad"]))
    if float(cfg.get("log_std_min", -20.0)) != -20.0:
        from stable_baselines3.sac import policies as _sacp
        _sacp.LOG_STD_MIN = float(cfg["log_std_min"])
    act = {"mish": torch.nn.Mish, "tanh": torch.nn.Tanh}[cfg.get("activation", "tanh")]
    model = DSRLSVGDino(
        "MlpPolicy", DummyVecEnv([lambda: env]), diffusion_policy=flow, diffusion_act_dim=(flow.horizon, flow.action_dim),
        noise_critic_grad_steps=cfg["noise_critic_steps"], critic_backup_combine_type="min", learning_rate=cfg["lr"],
        buffer_size=1, learning_starts=0, batch_size=cfg["batch_size"], tau=cfg["tau"],
        gamma=cfg["gamma"] ** cfg["action_steps"], train_freq=cfg["train_freq"], gradient_steps=cfg["utd"],
        ent_coef="auto", target_update_interval=1, target_entropy=cfg["target_entropy"], seed=seed,
        device=prior.device, verbose=0,
        policy_kwargs=dict(net_arch=dict(pi=net, qf=net), activation_fn=act, log_std_init=0.0,
                           post_linear_modules=[torch.nn.LayerNorm], n_critics=cfg["n_critics"],
                           share_features_extractor=False, **extractor))
    _, params, _ = load_from_zip_file(str(ckpt), device=prior.device)
    model.set_parameters(params, exact_match=True, device=prior.device)
    model.image_actor, model.frame_env = True, env
    model.deterministic_actor, model.eval_mode = bool(cfg.get("deterministic_actor")), True
    model.policy.set_training_mode(False)
    model.num_timesteps = 10 ** 12
    model.critic_kind = "q" if cfg.get("q_critic") else "dvd"
    if with_dvd and model.critic_kind == "dvd":
        attach_dvd(model, cfg, ckpt, prior.device, state_dim, flow)
    return model, env, flow, cfg


def attach_dvd(model, cfg, ckpt, device, state_dim, flow):
    """the DVD critic gamma^8 V(f(ctx, s, a)) exactly as dsrl_dino builds it, weights from the checkpoint's _svg.pt"""
    from svg_term import SvgTerm
    from online_rl_mimicgen import dino_svg
    from online_rl_mimicgen.value_model import ValueEnsemble
    from online_rl_mimicgen.pool_mqattn import MultiQueryAttnPool
    adapter = dino_svg.build(cfg["dino_wm_ckpt"], device=device, bf16=True, action_dim=7, arch=cfg["dino_wm_arch"], state_head=True)
    scaler = model.frame_env.scaler
    value_fn = ValueEnsemble(cfg["value_model"], cfg["value_variant"], scaler, state_dim=state_dim, device=device,
                             finetune=True, scratch=False)
    attn_pool = MultiQueryAttnPool.from_blob(torch.load(Path(cfg["value_model"]) / "mqattn_pool.pt", map_location=device,
                                                        weights_only=False), device=device)
    proprio_dim = dino_svg.resolve_proprio_dim(value_fn, cfg["dino_pool"])
    term = SvgTerm(value_fn, scaler, cfg["action_steps"] * flow.action_dim, device, discount=cfg["gamma"],
                   horizon=cfg["action_steps"], chunks=1, model_tag="dino_wm", finetune_dynamics=True,
                   dynamics_lr=cfg["svg_dynamics_lr"], dynamics=adapter, feature_space=True, pool_mode=cfg["dino_pool"],
                   value_pool_mode=None, proprio_dim=proprio_dim, fixed_scale=cfg.get("svg_fixed_scale", 0.0),
                   proprio_source=cfg["svg_proprio_source"], state_loss_weight=cfg.get("svg_state_loss_weight", 1.0))
    term._pool = attn_pool
    st = torch.load(str(ckpt).replace(".zip", "_svg.pt"), map_location=device, weights_only=False)
    value_fn.load_state_dict(st["value_fn"])
    missing, unexpected = adapter.load_state_dict(st["dynamics"], strict=False)
    assert not unexpected and all(k.startswith("wm.dino.") for k in missing), (unexpected, missing[:5])
    attn_pool.load_state_dict(st["attn_pool"])
    model.svg_term, model.value_fn, model.adapter, model.attn_pool = term, value_fn, adapter, attn_pool
    model.dvd_encode = lambda imgs: dino_svg.encode_frames(adapter, imgs)


# ------------------------------------------------------------------ exact snapshot (from the critic-ablation harness)
def _plain(v, depth=0):
    from robosuite.utils.buffers import Buffer
    if v is None or isinstance(v, (bool, int, float, str, complex, np.generic)):
        return True
    if isinstance(v, np.ndarray):
        return v.dtype != object
    if isinstance(v, Buffer):
        return True
    if depth < 4 and isinstance(v, (list, tuple, collections.deque)):
        return all(_plain(x, depth + 1) for x in v)
    if depth < 4 and isinstance(v, dict):
        return all(isinstance(k, (str, int)) and _plain(x, depth + 1) for k, x in v.items())
    return False


def _objects(raw):
    out = [(("env",), raw)]
    for i, rob in enumerate(raw.robots):
        out.append((("robot", i), rob))
        out.append((("robot", i, "controller"), rob.controller))
        for name in ("interpolator_pos", "interpolator_ori"):
            ip = getattr(rob.controller, name, None)
            if ip is not None:
                out.append((("robot", i, "controller", name), ip))
        grip = getattr(rob, "gripper", None)
        if grip is not None:
            out.append((("robot", i, "gripper"), grip))
    for name, obs in raw._observables.items():
        out.append((("observable", name), obs))
    return out


def fingerprint(raw) -> str:
    m = raw.sim.model._model
    h = hashlib.md5()
    for arr in (m.body_pos, m.body_quat, m.geom_pos, m.geom_size, m.qpos0):
        h.update(np.ascontiguousarray(arr).tobytes())
    h.update(np.array([m.nq, m.nv, m.nu, m.nbody, m.ngeom]).tobytes())
    return h.hexdigest()


def _resolve(raw, path):
    if path == ("env",):
        return raw
    if path[0] == "observable":
        return raw._observables[path[1]]
    obj = raw.robots[path[1]]
    for name in path[2:]:
        obj = getattr(obj, name)
    return obj


def snapshot(denv: DinoFrameEnv, observation: dict) -> bytes:
    import mujoco
    genv = denv.env
    raw = genv._env
    m, d = raw.sim.model._model, raw.sim.data._data
    spec = int(mujoco.mjtState.mjSTATE_INTEGRATION)
    state = np.empty(mujoco.mj_stateSize(m, spec))
    mujoco.mj_getState(m, d, state, spec)
    snap = dict(fp=fingerprint(raw), spec=spec, state=state,
                py={path: {k: copy.deepcopy(v) for k, v in vars(obj).items() if _plain(v)} for path, obj in _objects(raw)},
                wrapper=dict(_elapsed_steps=genv._elapsed_steps, _latest_images=copy.deepcopy(genv._latest_images)),
                observation=copy.deepcopy(observation),
                dino=dict(cur_frame=denv.cur_frame.copy(), prev_frame=None if denv.prev_frame is None else denv.prev_frame.copy(),
                          cur_pos=denv.cur_pos.copy(), primitive_steps=denv.primitive_steps))
    return zlib.compress(pickle.dumps(snap, protocol=5), 1)


def restore(denv: DinoFrameEnv, blob: bytes) -> np.ndarray:
    """exact restore; returns the SB3 observation vector at the snapshot"""
    import mujoco
    snap = pickle.loads(zlib.decompress(blob))
    genv = denv.env
    raw = genv._env
    if fingerprint(raw) != snap["fp"]:
        raise RuntimeError("snapshot belongs to another MjModel (reset the env with the episode's seed first)")
    m, d = raw.sim.model._model, raw.sim.data._data
    mujoco.mj_setState(m, d, snap["state"], snap["spec"])
    mujoco.mj_forward(m, d)
    mujoco.mj_setState(m, d, snap["state"], snap["spec"])
    for path, attrs in snap["py"].items():
        obj = _resolve(raw, path)
        for k, v in attrs.items():
            setattr(obj, k, copy.deepcopy(v))
    genv._elapsed_steps = snap["wrapper"]["_elapsed_steps"]
    genv._latest_images = copy.deepcopy(snap["wrapper"]["_latest_images"])
    dn = snap["dino"]
    denv.cur_frame, denv.prev_frame, denv.cur_pos = dn["cur_frame"].copy(), dn["prev_frame"], dn["cur_pos"].copy()
    denv.primitive_steps = dn["primitive_steps"]
    denv.ep_frames, denv.ep_actions = [], []
    return denv._observe(snap["observation"])


# ------------------------------------------------------------------ helpers
def seed_all(s: int):
    torch.manual_seed(s); np.random.seed(s); random.seed(s)


def act(model, obs_vec):
    model._last_obs = obs_vec[None]
    with torch.no_grad():
        action, _ = model._sample_action(learning_starts=0)
    return action[0]


def perturb(action_pad: np.ndarray, flow, m: float, axis: int) -> np.ndarray:
    a = action_pad.reshape(flow.horizon, flow.action_dim).copy()
    a[flow.executed, axis] = np.clip(a[flow.executed, axis] + m, -1.0, 1.0)
    return a.reshape(-1)


def q_heads(model, obs_vec, frame, actions) -> np.ndarray:
    """(K, n_critics) Q^A for K candidate actions at one state (no augmentation)"""
    dev = model.device
    k = len(actions)
    obs = torch.as_tensor(np.repeat(obs_vec[None], k, 0), device=dev, dtype=torch.float32)
    acts = torch.as_tensor(np.stack(actions), device=dev, dtype=torch.float32)
    imgs = torch.as_tensor(np.repeat(frame[None], k, 0), device=dev)
    with torch.no_grad():
        qs = model._q_forward(model.critic, obs, acts, imgs, False)
    return torch.cat(qs, dim=1).float().cpu().numpy()


def critic_values(model, flow, obs_vec, cur_frame, prev_frame, actions) -> np.ndarray:
    """(K, heads): Q^A heads for the Q arm, gamma^8 V(f) (ensemble mean, 1 column) for the DVD arm"""
    if model.critic_kind == "q":
        return q_heads(model, obs_vec, cur_frame, actions)
    dev = model.device
    k = len(actions)
    sd = model.frame_env.state_dim
    scaled = torch.as_tensor(np.repeat(obs_vec[None, :sd], k, 0), device=dev, dtype=torch.float32)
    chunks = torch.as_tensor(np.stack([a.reshape(flow.horizon, flow.action_dim)[flow.executed].reshape(-1) for a in actions]),
                             device=dev, dtype=torch.float32)
    prev = cur_frame if prev_frame is None else prev_frame
    with torch.no_grad():
        fp = model.dvd_encode(torch.as_tensor(prev[None], device=dev))
        fc = model.dvd_encode(torch.as_tensor(cur_frame[None], device=dev))
    ctx = torch.stack([fp.float(), fc.float()], dim=1).repeat(k, *([1] * (fp.dim())))
    return model._svg_scores(scaled, chunks, ctx).float().cpu().numpy()[:, None]


def capture_frames(denv, sink):
    """wrap the gym env's step so every primitive-step camera frame lands in sink"""
    genv = denv.env
    orig = genv.step

    def step(a):
        out = orig(a)
        px = out[0]["pixels"]
        sink.append({k: np.asarray(v).copy() for k, v in px.items()} if isinstance(px, dict) else np.asarray(px).copy())
        return out
    genv.step = step
    return orig


# ------------------------------------------------------------------ phases
def track_raw(env):
    """DinoFrameEnv keeps only derived fields; keep the raw observation dict too (snapshots store it)"""
    orig = env._track

    def _track(observation):
        env._snap_obs = observation
        orig(observation)
    env._track = _track


def phase_ref(a):
    model, env, flow, cfg = build(a.ckpt, a.seed, with_dvd=True)
    track_raw(env)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    seed_all(a.seed)
    env.episode_index = 0
    obs_vec, _ = env.reset()
    px0 = env._snap_obs["pixels"]
    frames: list = [{k: np.asarray(v).copy() for k, v in px0.items()} if isinstance(px0, dict) else np.asarray(px0).copy()]
    capture_frames(env, frames)
    recs, t, done, success = [], 0, False, False
    while not done:
        frame_idx = len(frames) - 1
        snap = snapshot(env, env._snap_obs)
        action = act(model, obs_vec)
        cands = [perturb(action, flow, m, a.axis) for m in a.deltas]
        q = critic_values(model, flow, obs_vec, env.cur_frame, env.prev_frame, cands)
        recs.append(dict(t=t, frame_idx=frame_idx, snap=snap, action=action, q=q, elapsed=int(env.env._elapsed_steps),
                         eef=env.cur_pos[:3].copy()))
        obs_vec, r, term, trunc, info = env.step(action)
        t += 1
        done, success = bool(term or trunc), bool(term)
    meta = dict(ckpt=str(a.ckpt), seed=a.seed, deltas=list(a.deltas), axis=a.axis, n_decisions=t, success=success,
                steps=int(env.primitive_steps), n_frames=len(frames), n_critics=int(q.shape[1]), critic=model.critic_kind)
    with open(out / "ref.pkl", "wb") as f:
        pickle.dump(dict(meta=meta, recs=recs), f, protocol=5)
    with open(out / "frames.pkl", "wb") as f:
        pickle.dump(frames, f, protocol=5)
    print(json.dumps(meta), flush=True)


def phase_gt(a):
    ref = pickle.load(open(Path(a.out) / "ref.pkl", "rb"))
    meta, recs = ref["meta"], ref["recs"]
    model, env, flow, cfg = build(Path(meta["ckpt"]), meta["seed"])
    track_raw(env)
    env.env.reset(seed=meta["seed"])                       # regenerate this episode's MjModel
    jobs = [(ti, di) for ti in range(0, len(recs), a.every) for di in range(len(meta["deltas"]))]
    jobs = jobs[a.worker::a.workers]
    res_dir = Path(a.out) / "gt"; res_dir.mkdir(exist_ok=True)
    for ti, di in jobs:
        f = res_dir / f"t{ti:03d}_d{di}.json"
        if f.exists():
            continue
        rec = recs[ti]
        cand = perturb(rec["action"], flow, meta["deltas"][di], meta["axis"])
        G, disp, ks, t0 = [], None, [], time.time()
        for n in range(a.n):
            obs_vec = restore(env, rec["snap"])
            seed_all(10 ** 6 * (ti + 1) + n)            # common random numbers: rollout n shares its seed across deltas
            pos0 = env.cur_pos[:3].copy()
            obs_vec, r, term, trunc, info = env.step(cand)
            if disp is None:
                disp = (env.cur_pos[:3] - pos0).tolist()
            k = 0
            while not (term or trunc):
                obs_vec, r, term, trunc, info = env.step(act(model, obs_vec))
                k += 1
            G.append(GAMMA8 ** k if term else 0.0); ks.append(k if term else -1)
        f.write_text(json.dumps(dict(t=ti, d=di, delta=meta["deltas"][di], G=G, k=ks, eef_disp_first_chunk=disp,
                                     seconds=time.time() - t0)))
        print(f"[gt w{a.worker}] t={ti} d={di}: Q_MC={np.mean(G):.3f} ({a.n} rollouts, {time.time() - t0:.0f}s)", flush=True)


def phase_score(a):
    ref = pickle.load(open(Path(a.out) / "ref.pkl", "rb"))
    meta, recs = ref["meta"], ref["recs"]
    model, env, flow, cfg = build(a.ckpt, meta["seed"], with_dvd=True)
    track_raw(env)
    env.env.reset(seed=meta["seed"])
    vals = []
    for rec in recs:
        obs_vec = restore(env, rec["snap"])
        cands = [perturb(rec["action"], flow, m, meta["axis"]) for m in meta["deltas"]]
        vals.append(critic_values(model, flow, obs_vec, env.cur_frame, env.prev_frame, cands))
    out = Path(a.out) / f"critic_{a.name}.pkl"
    pickle.dump(dict(ckpt=str(a.ckpt), kind=model.critic_kind, values=vals), open(out, "wb"))
    print("wrote", out, model.critic_kind, len(vals), flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("phase", choices=["ref", "gt", "score"])
    ap.add_argument("--name", default="other")
    ap.add_argument("--ckpt", type=Path)
    ap.add_argument("--seed", type=int, default=10000)
    ap.add_argument("--out", required=True)
    ap.add_argument("--axis", type=int, default=0, help="action dim offset (0/1/2 = x/y/z position command)")
    ap.add_argument("--deltas", type=float, nargs="+", default=[-0.03, -0.015, -0.006, 0.0, 0.006, 0.015, 0.03])
    ap.add_argument("--every", type=int, default=2)
    ap.add_argument("--n", type=int, default=16)
    ap.add_argument("--worker", type=int, default=0)
    ap.add_argument("--workers", type=int, default=1)
    a = ap.parse_args()
    {"ref": phase_ref, "gt": phase_gt, "score": phase_score}[a.phase](a)
