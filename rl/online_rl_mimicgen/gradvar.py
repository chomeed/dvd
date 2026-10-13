import torch


GRADVAR_EVERY = 500
GRADVAR_MICROBATCHES = 8


PHASE_NAMES = {1.0: "reach", 2.0: "grasp", 3.0: "neargoal"}


def _flat_grad(scalar, params, retain=True):
    grads = torch.autograd.grad(scalar, params, retain_graph=retain, allow_unused=True)
    return torch.cat([(torch.zeros_like(p) if g is None else g).reshape(-1)
                      for g, p in zip(grads, params)])


def gradient_snr(terms, residual, row_of_batch, batch_size, params, action=None,
                 composite=None, microbatches=GRADVAR_MICROBATCHES, phase=None,
                 out_grads=None):
    """Signal-to-noise statistics for the actor gradient, per loss term."""
    stats: dict[str, float] = {}
    sub_grads: dict[str, torch.Tensor] = {}
    term_rows: dict[str, float] = {}
    live = {name: t for name, t in terms.items()
            if isinstance(t, torch.Tensor) and t.requires_grad and t.grad_fn is not None}
    if not live:
        return stats


    m = min(microbatches, batch_size)
    assign = torch.randperm(batch_size, device=residual.device) % m
    masks = [(assign[row_of_batch] == j).float().unsqueeze(-1) for j in range(m)]

    full_grads: dict[str, torch.Tensor] = {}
    holder = {"mask": None}
    handle = residual.register_hook(
        lambda g: g if holder["mask"] is None else g * holder["mask"])
    try:
        for name, scalar in live.items():
            pieces = []
            for mask in masks:
                holder["mask"] = mask
                pieces.append(_flat_grad(scalar, params))
            holder["mask"] = None
            stacked = torch.stack(pieces)
            sub_grads[name] = stacked
            full = stacked.sum(0)
            full_grads[name] = full


            n_eff = float(len(row_of_batch))
            if action is not None and action.requires_grad:
                (per_sample,) = torch.autograd.grad(scalar, action, retain_graph=True,
                                                    allow_unused=True)
                if per_sample is not None:
                    row_norms = per_sample.norm(dim=-1)
                    live_rows = row_norms > 0
                    n_live = int(live_rows.sum())
                    if n_live:
                        n_eff = float(n_live)
                        kept = row_norms[live_rows]
                        stats[f"gradvar_agrad_rows_{name}"] = n_eff
                        stats[f"gradvar_agrad_norm_mean_{name}"] = float(kept.mean())
                        stats[f"gradvar_agrad_norm_std_{name}"] = float(kept.std())


                        stats[f"gradvar_agrad_ratio_{name}"] = float(
                            per_sample[live_rows].mean(0).norm() / kept.mean().clamp_min(1e-20))
                        stats[f"gradvar_agrad_floor_{name}"] = n_eff ** -0.5


                        if phase is not None:
                            for pid, pname in PHASE_NAMES.items():
                                sel = live_rows & (phase == pid)
                                n_p = int(sel.sum())
                                if n_p < 4:
                                    continue
                                pk = row_norms[sel]
                                stats[f"gradvar_agrad_rows_{name}_{pname}"] = float(n_p)
                                stats[f"gradvar_agrad_norm_mean_{name}_{pname}"] = float(pk.mean())
                                stats[f"gradvar_agrad_norm_std_{name}_{pname}"] = float(pk.std())
                                stats[f"gradvar_agrad_ratio_{name}_{pname}"] = float(
                                    per_sample[sel].mean(0).norm() / pk.mean().clamp_min(1e-20))
                                stats[f"gradvar_agrad_floor_{name}_{pname}"] = n_p ** -0.5
            term_rows[name] = n_eff
            stats.update(_stats_from_pieces(name, stacked, n_eff))
    finally:
        holder["mask"] = None
        handle.remove()

    if out_grads is not None:


        for _n, _g in sub_grads.items():
            out_grads[_n] = _g.sum(0).detach()


    if composite:
        parts = [(c, sub_grads[n]) for n, c in composite.items()
                 if n in sub_grads and c != 0.0]
        if parts:
            total = sum(c * g for c, g in parts)


            n_total = max(term_rows[n] for n, _ in composite.items() if n in term_rows)
            stats.update(_stats_from_pieces("actor", total, n_total))


    names = [n for n in ("q", "dvd", "bc", "actor") if n in full_grads]
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            if a == "actor" or b == "actor":
                continue
            stats[f"gradvar_cross_{a}_{b}"] = float(
                torch.nn.functional.cosine_similarity(full_grads[a], full_grads[b], dim=0))

    stats["gradvar_ratio_floor"] = float(m) ** -0.5
    stats["gradvar_microbatches"] = float(m)
    return stats


def _stats_from_pieces(name, stacked, n_rows):
    """Statistics from `stacked` = (m, P) MASKED sub-batch gradients that sum to the full one."""
    m = stacked.shape[0]
    out: dict[str, float] = {}
    full = stacked.sum(0)


    sub = stacked * float(m)
    norms = sub.norm(dim=1)

    unit = sub / norms.clamp_min(1e-20).unsqueeze(1)
    cosine = unit @ unit.T
    off = ~torch.eye(m, dtype=torch.bool, device=sub.device)
    out[f"gradvar_cos_{name}"] = float(cosine[off].mean())

    half_a, half_b = stacked[: m // 2].sum(0), stacked[m // 2:].sum(0)
    out[f"gradvar_cos_half_{name}"] = float(
        torch.nn.functional.cosine_similarity(half_a, half_b, dim=0))


    out[f"gradvar_ratio_{name}"] = float(full.norm() / norms.mean().clamp_min(1e-20))
    out[f"gradvar_gnorm_mean_{name}"] = float(norms.mean())
    out[f"gradvar_gnorm_std_{name}"] = float(norms.std())


    n_big = float(n_rows)
    n_small = n_big / m
    g_small_sq = float((norms ** 2).mean())
    g_big_sq = float(full.norm() ** 2)
    g_true_sq = (n_big * g_big_sq - n_small * g_small_sq) / (n_big - n_small)
    sigma = (g_small_sq - g_big_sq) / (1.0 / n_small - 1.0 / n_big)
    out[f"gradvar_noise_scale_{name}"] = (
        float(sigma / g_true_sq) if g_true_sq > 1e-20 else float("nan"))
    return out


def apply_flat_(params, flat, scale):
    """theta <- theta + scale * flat, in place."""
    i = 0
    with torch.no_grad():
        for prm in params:
            n = prm.numel()
            prm.add_(flat[i:i + n].view_as(prm), alpha=scale)
            i += n
    assert i == flat.numel(), f"layout mismatch: consumed {i} of {flat.numel()}"


def directional_return_probe(params, directions, eval_fn, epsilon, baseline_first=True):
    """J(theta + eps * d/||d||) for each named direction."""
    snapshot = [prm.detach().clone() for prm in params]

    def restore():
        with torch.no_grad():
            for prm, saved in zip(params, snapshot):
                prm.copy_(saved)

    out = {}
    if baseline_first:
        out["probe_return_base"] = float(eval_fn())
    for name, d in directions.items():
        if d is None:
            continue
        nrm = float(d.norm())
        if not (nrm > 0.0):
            continue
        apply_flat_(params, d, epsilon / nrm)
        try:
            out[f"probe_return_{name}"] = float(eval_fn())
        finally:
            restore()
        out[f"probe_dirnorm_{name}"] = nrm


    out["probe_return_zero"] = float(eval_fn())
    restore()
    return out
