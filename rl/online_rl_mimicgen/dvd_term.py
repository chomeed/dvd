"""DVD(1) actor gradient for train_online.py -- the follow-up the `dvd1` hook was left for."""
from __future__ import annotations
import os

import importlib
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


DYNAMICS_DIR = Path(os.environ.get("DVD_DYNAMICS_DIR", "dynamics_model"))


class DvdTerm:
    """gamma^H V(f(s, a)), normalised, plus optional online adaptation of f."""

    def __init__(self, value_fn, scaler, action_dim: int, device, discount: float = 0.99,
                 horizon: int = 8, chunks: int = 1, model_tag: str = "model_residual",
                 finetune_dynamics: bool = False, dynamics_lr: float = 1e-4,
                 dynamics_scratch: bool = False, dynamics=None,
                 value_aware_weight: float = 0.0,
                 identity_scaling: bool = False,
                 feature_space: bool = False, pool_mode: str = "mean", proprio_dim: int = 0,
                 value_pool_mode: str | None = None,
                 static_grad_block: str = "none", goal_slice: tuple[int, int] | None = None,
                 fixed_scale: float = 0.0,
                 proprio_source: str = "current", state_loss_weight: float = 1.0):
        self.value_fn = value_fn


        self.fixed_scale = float(fixed_scale)


        self._auto_n = int(-fixed_scale) if fixed_scale < 0 else 0
        self._auto_sum = 0.0
        self._auto_count = 0
        self._auto_frozen = None


        self.last_scale = float('nan')
        self.device = device


        self.feature_space = bool(feature_space)
        self.pool_mode = pool_mode
        self.proprio_dim = int(proprio_dim)


        self.value_pool_mode = value_pool_mode or pool_mode
        self._pool = None


        self.step2_policy = None


        self.last_step2_residual_norm = None


        self.chunks = max(1, int(chunks))
        self.discount = discount ** (horizon * self.chunks)


        self._gamma, self.horizon = discount, horizon
        self.finetune_dynamics = finetune_dynamics


        self.low = torch.as_tensor(np.asarray(scaler.low, np.float32), device=device)
        self.span = torch.as_tensor(np.asarray(scaler.scale, np.float32), device=device)


        self._static_mask = None
        if static_grad_block != "none":


            floor = 1e-3 if static_grad_block == "degenerate_wide" else 1e-5
            blocked = (self.span <= floor)
            n_degen = int(blocked.sum())
            if static_grad_block == "goal":
                if goal_slice is None:
                    raise ValueError("--dvd-static-grad goal needs a goal_slice for this task")
                blocked = blocked.clone()
                blocked[goal_slice[0]:goal_slice[1]] = True
            if blocked.any():
                self._static_mask = blocked.view(1, -1)
                print(f"[dvd] static-grad block '{static_grad_block}': "
                      f"{int(blocked.sum())} of {blocked.numel()} dims "
                      f"({n_degen} degenerate-span, spans "
                      f"{[round(float(x), 8) for x in self.span[blocked][:4]]}...); "
                      f"their actor gradient is zeroed, V's input value is unchanged", flush=True)
            else:
                print(f"[dvd] static-grad block '{static_grad_block}': no dims matched", flush=True)

        state_dim = int(self.low.numel())
        if self.feature_space:
            from online_rl_mimicgen.dino_wm import pool_features
            self._pool = pool_features
            if self.chunks > 2:
                raise ValueError(
                    f"--dvd-chunks {self.chunks} with a feature-space f: only k=1 and k=2 are "
                    "defined (chunk 2 runs the prior's ONE stored continuation / next_base).")

        if dynamics is not None:
            self.dynamics = dynamics
            if dynamics_scratch:
                raise ValueError("--dvd-dynamics-scratch is not defined for a pre-built f")
        else:
            self._load_dynamics(model_tag, state_dim, action_dim, device, dynamics_scratch)
        self.dynamics_scratch = dynamics_scratch
        self.value_aware_weight = float(value_aware_weight)


        self.identity_scaling = bool(identity_scaling)
        self.dynamics.requires_grad_(finetune_dynamics)
        self.dynamics.train(finetune_dynamics)
        self.optimiser = (torch.optim.Adam(self.dynamics.parameters(), lr=dynamics_lr)
                          if finetune_dynamics else None)


        self.generative = hasattr(self.dynamics, "mu_scale")


        self.anchored = hasattr(self.dynamics, "sample_anchored")


        self.unanchored_calls = 0
        self._warned_unanchored = False
        self._verified = False


        self.proprio_source = str(proprio_source)
        self.state_loss_weight = float(state_loss_weight)
        self.state_head_on = self.feature_space and bool(
            getattr(getattr(self.dynamics, "wm", None), "has_state_head", False))
        if self.proprio_source not in ("current", "predicted"):
            raise ValueError(f"proprio_source {self.proprio_source!r} is not 'current' or 'predicted'")
        if self.proprio_source == "predicted":
            if not (self.feature_space and self.proprio_dim):
                raise ValueError("--dvd-proprio-source predicted needs a feature-space V with a proprio tail")
            if not self.state_head_on:
                raise ValueError("--dvd-proprio-source predicted needs the world model's proprio head "
                                 "(--dino-wm-state-head)")
            if self.anchored or self.generative:
                raise ValueError("--dvd-proprio-source predicted is defined for the deterministic MSE f only")


        self.two_step_features = self.feature_space and self.chunks == 2
        if self.two_step_features:
            if not (self.state_head_on and self.proprio_source == "predicted"):
                raise ValueError(
                    "--dvd-chunks 2 with a feature-space f needs the world model's proprio head AND "
                    "--dvd-proprio-source predicted (pass --dino-wm-state-head --dvd-proprio-source "
                    "predicted): chunk 2 takes the chunk-1 PREDICTED proprio as its raw state, and V "
                    "reads the chunk-2 predicted proprio next to the chunk-2 features.")
            if not hasattr(self.dynamics, "ctx_feats"):
                raise ValueError("--dvd-chunks 2 with a feature-space f is wired for the DINO-WM adapter "
                                 "(ctx_feats context) only")
        if self.state_head_on:
            print(f"[dvd] proprio head ON: V proprio source={self.proprio_source}, residual loss "
                  f"weight={self.state_loss_weight} (trained={bool(finetune_dynamics)})", flush=True)
        print(f"[dvd] generative={self.generative} "
              f"dynamics={model_tag} finetune={finetune_dynamics} scratch={dynamics_scratch} "
              f"anchored={self.anchored} feature_space={self.feature_space} "
              f"pool={self.pool_mode if self.feature_space else '-'} "
              f"proprio_tail={self.proprio_dim if self.feature_space else '-'} "
              f"value={getattr(value_fn, 'variant', '?')} action_dim={action_dim} "
              f"discount^{horizon}={self.discount:.4f}", flush=True)

    def _load_dynamics(self, model_tag, state_dim, action_dim, device, dynamics_scratch):
        sys.path.insert(0, str(DYNAMICS_DIR))


        module = next((m for m in ("model_gaussian", "model_flow", "model_history",
                                   "model_residual", "model_rnn")
                       if model_tag.startswith(m)), "model")
        payload = torch.load(DYNAMICS_DIR / f"dynamics_{model_tag}.pt", map_location=device)

        if isinstance(payload, dict) and "state_dict" in payload:
            state_dim, action_dim = payload["state_dim"], payload["chunk_dim"]
            payload = payload["state_dict"]


        first = next((v for k, v in payload.items() if k.endswith("net.0.weight")), None)
        kwargs = {"hidden": int(first.shape[0])} if first is not None else {}


        mu = payload.get("mu")


        if "freqs" in payload:
            mu = None
        if mu is not None and mu.dim() == 1 and first is not None:
            ck_state = int(mu.shape[0])
            ck_action = int(first.shape[1]) - ck_state
            if ck_state != state_dim or ck_action != action_dim:
                print(f"[dvd] dynamics checkpoint is {ck_state}x{ck_action}, not the task's "
                      f"{state_dim}x{action_dim} -- sizing f from the checkpoint", flush=True)
                state_dim, action_dim = ck_state, ck_action
        self.dynamics = importlib.import_module(module).Dynamics(
            state_dim, action_dim, **kwargs).to(device)


        self.dynamics.load_state_dict(payload)
        if dynamics_scratch:
            for module in self.dynamics.modules():
                if hasattr(module, "reset_parameters"):
                    module.reset_parameters()


    def value_input(self, predicted: torch.Tensor, scaled_state: torch.Tensor,
                    proprio: torch.Tensor | None = None) -> torch.Tensor:
        """f's output, in V's input space."""
        if not self.feature_space:
            return self.scaled_state(self.block_static_grad(predicted))
        pooled = self._pool(predicted, self.value_pool_mode)
        if not self.proprio_dim:
            return pooled
        if proprio is None:
            proprio = self.raw_state(scaled_state)[:, :self.proprio_dim]
        return torch.cat([pooled, proprio[:, :self.proprio_dim]], dim=-1)

    def model_proprio(self, predicted: torch.Tensor, record: bool = False) -> torch.Tensor | None:
        """The proprio V reads next to f's predicted t+8 features:"""
        if self.proprio_source != "predicted":
            return None
        p = self.dynamics.last_state_pred
        if p is None or p.shape[0] != predicted.shape[0]:
            raise RuntimeError("model_proprio: the proprio head has no output for this batch "
                               f"(got {None if p is None else tuple(p.shape)}, features {tuple(predicted.shape)})")
        if record:
            wm = self.dynamics.wm


            self._shift_rows = (wm.last_state_delta[:, :3] * wm.state_delta_scale[:3]).detach().norm(dim=-1)
            self.last_proprio_shift_mm = float(self._shift_rows.mean() * 1000)
        return p[:, :self.proprio_dim]

    def block_static_grad(self, predicted: torch.Tensor) -> torch.Tensor:
        """Stop the actor gradient on state dims the action cannot move."""
        if self._static_mask is None:
            return predicted
        return torch.where(self._static_mask, predicted.detach(), predicted)


    def raw_state(self, scaled: torch.Tensor) -> torch.Tensor:
        if self.identity_scaling:
            return scaled
        return (scaled + 1.0) * 0.5 * self.span + self.low

    def raw_env_state(self, scaled: torch.Tensor) -> torch.Tensor:
        """Unscale the TASK state, ALWAYS -- regardless of identity_scaling."""
        return (scaled + 1.0) * 0.5 * self.span + self.low

    def scaled_state(self, raw: torch.Tensor) -> torch.Tensor:
        if self.identity_scaling:
            return raw
        return torch.clamp(2.0 * (raw - self.low) / self.span - 1.0, -5.0, 5.0)


    def objective(self, scaled_state: torch.Tensor, action: torch.Tensor,
                  follow: torch.Tensor | None = None,
                  anchor: tuple[torch.Tensor, torch.Tensor] | None = None,
                  row_chunks: int = 1):
        """(normalised objective to MINIMISE, mean value for logging)."""
        if not self._verified:
            self.verify(scaled_state)
            self._verified = True
        was_training = self.dynamics.training
        self.dynamics.eval()
        if int(row_chunks) > 1:
            try:
                value = self._chunked_value(scaled_state, action, follow, anchor, int(row_chunks))
            finally:
                self.dynamics.train(was_training)
            scale = self._norm(value)
            return -value.mean() / scale, float(value.detach().mean())
        predicted_raw = self._rollout(scaled_state, action, follow, anchor)


        value = self.discount * self.value_fn(self.value_input(
            predicted_raw, scaled_state,
            self.model_proprio(predicted_raw, record=not self.two_step_features)))
        self.dynamics.train(was_training)
        scale = self._norm(value)
        return -value.mean() / scale, float(value.detach().mean())

    def lookahead_discount(self, chunks: int) -> float:
        """gamma^(H * chunks)."""
        return self._gamma ** (self.horizon * int(chunks))

    def feature_step2(self, predicted: torch.Tensor, proprio: torch.Tensor,
                      follow: torch.Tensor, action: torch.Tensor | None = None) -> torch.Tensor:
        """Chunk 2 of the DINO world model:"""
        adapter = self.dynamics
        saved = adapter.ctx_feats
        if saved is None:
            raise RuntimeError("feature_step2: adapter.ctx_feats is None -- install the [t-8, t] context first")
        if saved.shape[0] != predicted.shape[0]:
            raise RuntimeError(f"feature_step2: context rows {saved.shape[0]} != predicted rows {predicted.shape[0]}")
        try:
            adapter.ctx_feats = torch.stack([saved[:, -1], predicted.to(saved.dtype)], dim=1)
            return adapter(proprio, follow.detach() if action is None else action)
        finally:
            adapter.ctx_feats = saved

    def _feature_continue(self, predicted: torch.Tensor, follow: torch.Tensor | None) -> torch.Tensor:
        """DVD(k=2) in feature space:"""
        if follow is None:
            raise ValueError("feature-space DVD(k=2) needs a `follow` chunk (the recorded next_base or "
                             "the prior's continuation); repeating the optimised action would put a second "
                             "gradient path through it")
        self.model_proprio(predicted, record=True)
        if self.step2_policy is None:
            return self.feature_step2(predicted, self.dynamics.last_state_pred, follow)


        proprio1 = self.dynamics.last_state_pred
        action2 = self.step2_policy(predicted, proprio1, follow.detach())
        return self.feature_step2(predicted, proprio1, follow, action=action2)

    def _norm(self, t: torch.Tensor) -> torch.Tensor:
        """Denominator for the DVD term."""
        if self.fixed_scale > 0.0:
            self.last_scale = float(self.fixed_scale)
            return torch.as_tensor(self.fixed_scale, dtype=t.dtype, device=t.device)
        batch = t.detach().abs().mean().clamp_min(1e-3)
        if self._auto_n > 0:
            if self._auto_frozen is not None:
                self.last_scale = float(self._auto_frozen)
                return torch.as_tensor(self._auto_frozen, dtype=t.dtype, device=t.device)
            self._auto_sum += float(batch); self._auto_count += 1
            self.last_scale = float(batch)
            if self._auto_count >= self._auto_n:
                self._auto_frozen = max(self._auto_sum / self._auto_count, 1e-3)
                print(f"[dvd-scale] auto-calibrated to {self._auto_frozen:.4f} "
                      f"over {self._auto_count} updates; frozen for the rest of the run", flush=True)
        self.last_scale = float(batch)
        return batch

    def _warn_unanchored(self) -> None:
        """Loud, once:"""
        self.unanchored_calls += 1
        if self._warned_unanchored:
            return
        self._warned_unanchored = True
        print("\n" + "!" * 96 + "\n"
              "[dvd] UNANCHORED ROLLOUT ON AN ANCHORED f -- THIS ACTOR GRADIENT IS WRONG\n"
              "  self.anchored is True (f exposes sample_anchored) but _rollout was called with\n"
              "  anchor=None, so it fell back to self.dynamics(s, a) = forward(). For a flow that\n"
              "  integrates from FRESH NOISE every call: the residual no longer cancels and d/da\n"
              "  goes through resampled noise, not the Jacobian at fixed xi. Every DVD metric\n"
              "  will still look healthy -- that is the whole hazard.\n"
              "  The caller that does this today is objective_terms() (--dvd-phase-gate), which\n"
              "  takes an `anchor` argument and is not handed one by train_online.py:1145.\n"
              "  Counted in dvd_term.unanchored_calls; this banner prints ONCE per run.\n"
              + "!" * 96 + "\n", file=sys.stderr, flush=True)

    def _rollout(self, scaled_state: torch.Tensor, action: torch.Tensor,
                 follow: torch.Tensor | None,
                 anchor: tuple[torch.Tensor, torch.Tensor] | None = None):
        """f applied `self.chunks` times."""
        raw_state = self.raw_state(scaled_state)
        if self.anchored and anchor is None:


            self._warn_unanchored()
        if self.anchored and anchor is not None:


            action_data, next_target = anchor


            raw = self.dynamics.sample_anchored(
                raw_state, action, action_data,
                next_target if self.feature_space else self.raw_state(next_target))
        else:
            raw = self.dynamics(raw_state, action)
        if self.two_step_features:


            return self._feature_continue(raw, follow)
        nxt = action if follow is None else follow
        for _ in range(self.chunks - 1):


            raw = self.dynamics(raw, nxt)
        return raw


    def row_chunk_refusal(self) -> str | None:
        """Why this term cannot be row-chunked exactly, or None."""
        if not (self.feature_space and hasattr(self.dynamics, "ctx_feats")):
            return ("--dvd-row-chunks is wired for the DINO-WM feature path only (--dvd-dynamics dino_wm): "
                    "the chunks slice the adapter's ctx_feats context")
        if self.anchored or self.generative:
            return ("--dvd-row-chunks with an anchored/generative f (a flow head): refused -- the anchored "
                    "rollout was not verified under chunking")
        if self.step2_policy is not None:
            return ("--dvd-row-chunks with --dvd-step2-action actor: the step-2 residual puts actor-parameter "
                    "gradient into the objective through a path other than the action rows, which the "
                    "action-row surrogate cannot carry")
        if hasattr(self.value_fn, "arm_scale_update"):
            return ("--dvd-row-chunks with a DualValue (--value-td-model): its forward normalises by a BATCH "
                    "statistic and updates its EMA scales on the first >= MIN_ROWS call, so a row's value "
                    "depends on which chunk it is in")
        mods = [self.value_fn, self.dynamics]
        if isinstance(self._pool, torch.nn.Module):
            mods.append(self._pool)
        for top in mods:
            if not isinstance(top, torch.nn.Module):
                continue
            for name, m in top.named_modules():
                if not m.training:
                    continue
                if (isinstance(m, torch.nn.modules.dropout._DropoutNd) and m.p > 0) or \
                        (isinstance(m, torch.nn.MultiheadAttention) and m.dropout > 0):
                    return (f"--dvd-row-chunks with an ACTIVE dropout ({type(top).__name__}.{name} in train "
                            "mode): the chunked passes would draw different masks than one full pass")
        return None

    def _chunked_value(self, scaled_state: torch.Tensor, action: torch.Tensor,
                       follow: torch.Tensor | None, anchor, n: int) -> torch.Tensor:
        """gamma^H V(f(s, a)) for all R rows, evaluated over `n` contiguous row chunks -> (R, 1)."""
        why = self.row_chunk_refusal()
        if why is not None:
            raise ValueError(why)
        adapter = self.dynamics
        saved = adapter.ctx_feats
        rows = int(action.shape[0])
        if saved is None or int(saved.shape[0]) != rows:
            raise RuntimeError(f"--dvd-row-chunks: the installed context has "
                               f"{None if saved is None else int(saved.shape[0])} rows, the objective {rows}")
        slices, start = [], 0
        for part in torch.arange(rows).tensor_split(min(int(n), rows)):
            if part.numel():
                slices.append(slice(start, start + int(part.numel())))
                start += int(part.numel())
        need_grad = torch.is_grad_enabled() and action.requires_grad
        if torch.is_grad_enabled() and (scaled_state.requires_grad
                                        or (follow is not None and follow.requires_grad)):


            raise ValueError("--dvd-row-chunks: the DVD state / follow rows carry gradient; only the action rows "
                             "may (use --dvd-row-chunks 1 for this configuration)")
        wm = getattr(adapter, "wm", None)
        state_head = wm is not None and getattr(wm, "has_state_head", False)
        values, grads, deltas, preds, shifts = [], [], [], [], []
        try:
            for sl in slices:
                adapter.ctx_feats = saved[sl]
                a = action[sl].detach()
                if need_grad:
                    a.requires_grad_(True)
                self._shift_rows = None
                st = scaled_state[sl]
                predicted_raw = self._rollout(st, a, None if follow is None else follow[sl], None)
                v = self.discount * self.value_fn(self.value_input(
                    predicted_raw, st,
                    self.model_proprio(predicted_raw, record=not self.two_step_features)))
                if need_grad:


                    g = (torch.autograd.grad(v.sum(), a, allow_unused=True)[0] if v.requires_grad else None)
                    grads.append(torch.zeros_like(a) if g is None else g)
                values.append(v.detach())
                if self._shift_rows is not None:
                    shifts.append(self._shift_rows)
                if state_head and getattr(wm, "last_state_delta", None) is not None:
                    deltas.append(wm.last_state_delta.detach())
                    preds.append(wm.last_state_pred.detach())
                del predicted_raw, v, a
        finally:
            adapter.ctx_feats = saved
        if state_head and len(deltas) == len(slices):
            wm.last_state_delta = torch.cat(deltas)
            wm.last_state_pred = torch.cat(preds)
        if shifts and len(shifts) == len(slices):
            self._shift_rows = torch.cat(shifts)
            self.last_proprio_shift_mm = float(self._shift_rows.mean() * 1000)
        value = torch.cat(values)
        if need_grad:
            g = torch.cat(grads)
            x = (action * g).flatten(1).sum(dim=-1, keepdim=True).to(value.dtype)
            value = value + (x - x.detach()).view_as(value)
        return value

    def objective_terms(self, scaled_state: torch.Tensor, action: torch.Tensor,
                        follow: torch.Tensor | None = None,
                        anchor: tuple[torch.Tensor, torch.Tensor] | None = None,
                        row_chunks: int = 1):
        """Per-sample -gamma^H V(f(s,a)) / |.|, i.e."""
        if not self._verified:
            self.verify(scaled_state)
            self._verified = True
        was_training = self.dynamics.training
        self.dynamics.eval()
        if int(row_chunks) > 1:
            try:
                value = self._chunked_value(scaled_state, action, follow, anchor, int(row_chunks))
            finally:
                self.dynamics.train(was_training)
            scale = self._norm(value)
            return -value / scale, float(value.detach().mean())
        predicted_raw = self._rollout(scaled_state, action, follow, anchor)
        value = self.discount * self.value_fn(self.value_input(
            predicted_raw, scaled_state,
            self.model_proprio(predicted_raw, record=not self.two_step_features)))
        self.dynamics.train(was_training)
        scale = self._norm(value)
        return -value / scale, float(value.detach().mean())


    def replay_objective(self, actor, batch, residual_scale, weight_clip=10.0):
        """The paper's line 13, minus r_a."""
        if self.feature_space and "next_feat_obs" not in batch:


            raise NotImplementedError(
                "--dvd-mode replay on a feature-space f needs batch['next_feat_obs'] (the encoded "
                "t+8 features). Pass --vtrace/replay through train_online, which supplies it.")
        if self.proprio_source == "predicted":
            raise NotImplementedError("--dvd-mode replay with --dvd-proprio-source predicted: xi_k would "
                                      "need a proprio residual too; not implemented")
        state, base = batch["state"], batch["base"]
        replayed =((batch["action"] - base) / residual_scale).clamp(-1.0 + 1e-6, 1.0 - 1e-6)

        mean, log_std = actor(state, base)
        std = log_std.exp()
        eta = ((torch.atanh(replayed) - mean) / std).detach()
        residual_theta = torch.tanh(mean + std * eta)
        action_theta = (base + residual_scale * residual_theta).clamp(-1.0, 1.0)

        raw_state = self.raw_state(state)


        raw_next = (batch["next_feat_obs"] if self.feature_space
                    else self.raw_state(batch.get("next_state_1step", batch["next_state"])))
        was_training = self.dynamics.training
        self.dynamics.eval()
        if self.generative:


            with torch.no_grad():
                xi = self.dynamics.infer_noise(raw_state, batch["action"], raw_next)


                clipped = (xi.abs() > 5.0).float().mean()
                xi = xi.clamp(-5.0, 5.0)
            mean, scale = self.dynamics.mu_scale(raw_state, action_theta)
            predicted = mean + scale * xi
        else:
            with torch.no_grad():
                xi = raw_next - self.dynamics(raw_state, batch["action"])
                clipped = torch.zeros((), device=xi.device)
            predicted = self.dynamics(raw_state, action_theta) + xi
        self.dynamics.train(was_training)
        value = self.discount * self.value_fn(self.value_input(predicted, state))

        with torch.no_grad():
            logp_now = actor.log_prob_of(state, base, replayed)
            behaviour_logp = batch["behaviour_logp"]


            valid = torch.isfinite(behaviour_logp)
            ratio = torch.where(
                valid,
                torch.exp(logp_now - torch.nan_to_num(behaviour_logp)).clamp(max=weight_clip),
                torch.ones_like(logp_now),
            )
        weighted = ratio * value
        norm = self._norm(weighted)
        return (-weighted.mean() / norm, float(value.detach().mean()),
                float(ratio.mean()), float(valid.float().mean()), float(clipped))


    def fit_step(self, batch: dict) -> dict[str, float]:
        """One supervised step on the agent's OWN transitions, so f tracks the policy's states."""
        if not self.finetune_dynamics:
            return {}
        if getattr(self.dynamics, "is_latent56", False):
            return self._fit_step_latent(batch)
        if self.feature_space:
            return self._fit_step_features(batch)
        raw_state = self.raw_state(batch["state"]).detach()


        raw_next = self.raw_state(
            batch.get("next_state_1step", batch["next_state"])).detach()
        action = batch["action"].detach()
        self.optimiser.zero_grad()
        loss = self.dynamics.loss(raw_state, action, raw_next)


        vaml = None
        if self.value_aware_weight > 0.0:
            pred_s = self.dynamics(raw_state, action)
            with torch.no_grad():
                v_target = self.value_fn(self.value_input(raw_next, None))
            v_pred = self.value_fn(self.value_input(pred_s, None))
            vaml = torch.nn.functional.mse_loss(v_pred, v_target)
            loss = loss + self.value_aware_weight * vaml
        loss.backward()
        self.optimiser.step()
        with torch.no_grad():
            predicted = self.dynamics(raw_state, action)
            if self.identity_scaling:


                error = None
                feat_mse = F.mse_loss(predicted, raw_next)
            else:


                error = (predicted - raw_next)[:, 9:12].norm(dim=-1).mean() * 1000
                feat_mse = None
        out = {"dvd_dynamics_loss": float(loss.detach())}
        if error is not None:
            out["dvd_pod_mm"] = float(error)
        else:
            out["dvd_feat_mse"] = float(feat_mse)
        if vaml is not None:
            out["dvd_vaml"] = float(vaml.detach())
        return out

    def _fit_step_latent(self, batch: dict) -> dict[str, float]:
        """One supervised step on the 56-d latent world model."""
        action = batch["action"].detach()
        target = batch["target_feat"].detach()
        self.optimiser.zero_grad()
        pred = self.dynamics(None, action)
        loss = F.mse_loss(pred, target)


        import os as _os
        if _os.environ.get("LAT_DUMP") and not getattr(self, "_lat_dumped", False):
            self._lat_dumped = True
            torch.save({"prev_z": self.dynamics.prev_z, "cur_z": self.dynamics.cur_z,
                        "prop_n": self.dynamics.prop_n, "action": action, "target": target,
                        "pred": pred.detach(), "data_source": batch.get("data_source")},
                       _os.environ["LAT_DUMP"])
            print(f"[lat-dump] wrote first fit batch -> {_os.environ['LAT_DUMP']}", flush=True)
        loss.backward()
        self.optimiser.step()
        with torch.no_grad():

            copy_mse = F.mse_loss(self.dynamics.cur_z[:len(target)], target)
            feat_mse = F.mse_loss(self.dynamics(None, action), target)
        out = {"dvd_dynamics_loss": float(loss.detach()), "dvd_feat_mse": float(feat_mse),
               "dvd_copy_feat_mse": float(copy_mse)}
        if float(copy_mse) > 0:
            out["dvd_skill_vs_copy"] = 1.0 - float(feat_mse) / float(copy_mse)
        return out

    def _fit_step_features(self, batch: dict) -> dict[str, float]:
        """One supervised step of the world model against the ENCODER's own t+8 features."""
        raw_state = self.raw_state(batch["state"]).detach()
        action = batch["action"].detach()
        target = batch["target_feat"].detach()
        self.optimiser.zero_grad()
        loss = self.dynamics.loss(raw_state, action, target)
        state_loss, next_proprio = None, None
        if self.state_head_on:


            _nx = batch.get("next_state_1step", batch["next_state"]).detach()
            if _nx.shape[-1] != self.low.numel():
                raise RuntimeError(f"proprio head target: next state has width {_nx.shape[-1]}, "
                                   f"not the env state's {self.low.numel()}")
            next_proprio = self.raw_state(_nx)
            state_loss = self.dynamics.state_residual_loss(raw_state, next_proprio)
            loss = loss + self.state_loss_weight * state_loss


        vaml = None
        if self.value_aware_weight > 0.0:


            pred_f = getattr(self.dynamics, "_last_pred", None)
            if pred_f is None:
                pred_f = self.dynamics(raw_state, action)


            _vp = [q for q in self.value_fn.parameters() if q.requires_grad]
            for q in _vp:
                q.requires_grad_(False)
            try:
                k = min(len(raw_state), 8)
                sl = slice(0, k)
                _pp = self.model_proprio(pred_f)
                with torch.no_grad():
                    v_target = self.value_fn(self.value_input(
                        target[sl], batch["state"][sl],
                        None if _pp is None else next_proprio[sl, :self.proprio_dim]))
                v_pred = self.value_fn(self.value_input(
                    pred_f[sl], batch["state"][sl], None if _pp is None else _pp[sl]))
                vaml = F.mse_loss(v_pred, v_target)
            finally:
                for q in _vp:
                    q.requires_grad_(True)
            loss = loss + self.value_aware_weight * vaml
        loss.backward()
        self.optimiser.step()
        with torch.no_grad():
            error = F.mse_loss(self.dynamics(raw_state, action), target)


            state_metrics = (self.state_head_metrics(raw_state, next_proprio)
                             if self.state_head_on else {})
            copy = self.dynamics._context(raw_state)[:, -1]
            copy_mse = F.mse_loss(copy, target)
        out = {"dvd_dynamics_loss": float(loss.detach()), "dvd_feat_mse": float(error),
               "dvd_copy_feat_mse": float(copy_mse), **state_metrics}
        if state_loss is not None:
            out["dvd_state_loss"] = float(state_loss.detach())
        if vaml is not None:
            out["dvd_vaml"] = float(vaml.detach())
        if float(copy_mse) > 0:
            out["dvd_skill_vs_copy"] = 1.0 - float(error) / float(copy_mse)
        return out

    def probe_features(self, batch: dict) -> dict[str, float]:
        """_fit_step_features' post-step metrics on `batch`, WITHOUT the step:"""
        with torch.no_grad():
            raw_state = self.raw_state(batch["state"])
            action = batch["action"]
            target = batch["target_feat"]
            error = F.mse_loss(self.dynamics(raw_state, action), target)
            state_metrics = {}
            if self.state_head_on:
                _nx = batch.get("next_state_1step", batch["next_state"])
                if _nx.shape[-1] == self.low.numel():
                    state_metrics = self.state_head_metrics(raw_state, self.raw_state(_nx))
            copy_mse = F.mse_loss(self.dynamics._context(raw_state)[:, -1], target)
        out = {"dvd_feat_mse": float(error), "dvd_copy_feat_mse": float(copy_mse), **state_metrics}
        if float(copy_mse) > 0:
            out["dvd_skill_vs_copy"] = 1.0 - float(error) / float(copy_mse)
        return out

    @torch.no_grad()
    def state_head_metrics(self, raw_state: torch.Tensor, next_proprio: torch.Tensor) -> dict[str, float]:
        """Proprio head accuracy on the LAST forward's rows:"""
        wm = self.dynamics.wm
        delta, pred = wm.last_state_delta, wm.last_state_pred
        k = delta.shape[-1]
        tgt = (next_proprio[:, :k] - raw_state[:, :k]) / wm.state_delta_scale
        mse, copy = F.mse_loss(delta, tgt), tgt.pow(2).mean()
        return {"dvd_state_mse_n": float(mse), "dvd_state_copy_mse_n": float(copy),
                "dvd_state_skill": 1.0 - float(mse) / max(float(copy), 1e-12),
                "dvd_eef_mm": float((pred[:, :3] - next_proprio[:, :3]).norm(dim=-1).mean() * 1000),
                "dvd_eef_copy_mm": float((raw_state[:, :3] - next_proprio[:, :3]).norm(dim=-1).mean() * 1000)}


    def verify(self, scaled_state: torch.Tensor) -> None:
        if self.identity_scaling:


            print(f"[dvd] identity scaling (encoder space, dim {scaled_state.shape[-1]}); "
                  f"unscale check skipped", flush=True)
            return
        raw = self.raw_state(scaled_state)
        error = (self.scaled_state(raw) - scaled_state).abs().max().item()
        assert error < 1e-3, f"state unscaling round-trip off by {error}"


        floor = 0.5 * max(self.low.abs().max().item(),
                          (self.low + self.span).abs().max().item())
        assert raw.abs().max().item() > floor, (
            f"unscaled state peaks at {raw.abs().max().item():.3f}, below {floor:.3f} = half the "
            "demo range implied by the scaler -- StateScaler assumptions changed")
        print(f"[dvd] unscale verified (round-trip {error:.2e}, raw range "
              f"[{raw.min().item():.3f}, {raw.max().item():.3f}], floor {floor:.3f})", flush=True)
