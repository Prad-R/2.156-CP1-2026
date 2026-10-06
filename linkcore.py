"""Shared pieces for the optimizing stages: packing, fronts, Adam, alignment.

All mechanisms here are dicts in solver order (ground 0, crank 1, other
grounds, then moving joints each after both of its parents) with the traced
joint last, which is what screen.py, refine*.py and grow.py produce.
"""
import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import numpy as np
import jax
import jax.numpy as jnp

from LINKS.Geometry._utils import equisample, find_optimal_correspondences
from LINKS.Kinematics._solvers import dyadic_solve
from LINKS.Optimization._utils import material, unscaled_distance

MAX_N, T, CHUNK = 20, 200, 256
THETAS = np.linspace(0.0, 2 * np.pi, T)
B1, B2, EPS = 0.9, 0.999, 1e-8


def with_x(m, x):
    return {**m, "x0": np.array(x[: len(m["x0"])], dtype=np.float64)}


def pack(mechs):
    n = len(mechs)
    x0 = np.zeros((n, MAX_N, 2))
    A = np.zeros((n, MAX_N, MAX_N))
    types = np.ones((n, MAX_N, 1))
    tidx = np.zeros(n, dtype=np.int32)
    for i, m in enumerate(mechs):
        k = len(m["x0"])
        x0[i, :k] = m["x0"]
        A[i, m["edges"][:, 0], m["edges"][:, 1]] = 1.0
        A[i, m["edges"][:, 1], m["edges"][:, 0]] = 1.0
        types[i, :k, 0] = 0.0
        types[i, m["fixed_joints"], 0] = 1.0
        tidx[i] = m["target_joint"]
    return x0, A, types, tidx


def batched(fn, mechs, n_out, *extra):
    """Run a jitted kernel over mechs in fixed-size chunks (avoids re-jitting)."""
    outs = [[] for _ in range(n_out)]
    for c in range(0, len(mechs), CHUNK):
        chunk = mechs[c:c + CHUNK]
        res = fn(*pack(chunk + [chunk[0]] * (CHUNK - len(chunk))), *extra)
        res = res if isinstance(res, tuple) else (res,)
        for o, r in zip(outs, res):
            o.append(np.array(r)[: len(chunk)])
    return [np.concatenate(o) for o in outs]


# ---------------------------------------------------------------- kernels
def normalized_pair(x0s, As, types, tidx, tgt):
    """Traced curve and target, resampled, centered and scaled exactly as the scorer does."""
    sol = dyadic_solve(As, x0s, types, THETAS)
    bad = jnp.isnan(sol).any(axis=(1, 2, 3))
    c = sol[jnp.arange(sol.shape[0]), tidx]
    c = jnp.nan_to_num(c * (~bad)[:, None, None], nan=0.0) + tgt * bad[:, None, None]
    nt, nc = equisample(tgt, T), equisample(c, T)
    nt, nc = nt - nt.mean(1, keepdims=True), nc - nc.mean(1, keepdims=True)
    s = jnp.sqrt((nt ** 2).sum(-1).sum(-1) / T)[:, None, None]
    return nc / s, nt / s, bad


@jax.jit
def score(x0s, As, types, tidx, target):
    """Official distance and material."""
    _, d = unscaled_distance(x0s, As, types, THETAS, tidx, jnp.broadcast_to(target, (x0s.shape[0], T, 2)))
    _, m = material(x0s, As)
    return d, m


@jax.jit
def value_and_grad(x0s, As, types, tidx, w, target):
    """Gradient of  official distance + w * material; also returns distance and material."""
    tgt = jnp.broadcast_to(target, (x0s.shape[0], T, 2))

    def loss(x):
        _, d = unscaled_distance(x, As, types, THETAS, tidx, tgt)
        _, m = material(x, As)
        return (jnp.where(jnp.isfinite(d), d, 0.0) + w * m).sum(), (d, m)
    (_, (d, m)), g = jax.value_and_grad(loss, has_aux=True)(x0s)
    return d, m, g


@jax.jit
def align(x0s, As, types, tidx, target):
    """Aligned traced curve, normalized target, per-point error and distance."""
    nc, nt, bad = normalized_pair(x0s, As, types, tidx, jnp.broadcast_to(target, (x0s.shape[0], T, 2)))
    _, aligned, d = find_optimal_correspondences(nc, nt, return_rotation_matrices=False,
                                                 return_aligned_curves=True, return_distances=True)
    err = jnp.linalg.norm(aligned - nt, axis=-1)
    return aligned, nt, err, jnp.where(bad, jnp.inf, d)


@jax.jit
def traced_radius(x0s, As, types, tidx):
    """rms radius of each traced curve (nan if the mechanism does not assemble)."""
    sol = dyadic_solve(As, x0s, types, THETAS)
    ok = jnp.isfinite(sol).all(axis=(1, 2, 3))
    c = sol[jnp.arange(sol.shape[0]), tidx]
    circle = jnp.stack([jnp.cos(THETAS), jnp.sin(THETAS)], -1).astype(c.dtype)
    c = equisample(jnp.where(ok[:, None, None], c, circle), T)
    c = c - c.mean(1, keepdims=True)
    return jnp.where(ok, jnp.sqrt((c ** 2).sum(-1).mean(-1)), jnp.nan)


def target_radius(target):
    t = np.array(equisample(jnp.asarray(target[None]), T))[0]
    return float(np.sqrt(((t - t.mean(0)) ** 2).sum(-1).mean()))


# ---------------------------------------------------------------- fronts
def nondominated(F):
    order = np.lexsort((F[:, 1], F[:, 0]))
    best_m = np.minimum.accumulate(F[order, 1])
    keep = np.ones(len(F), dtype=bool)
    keep[1:] = F[order, 1][1:] < best_m[:-1]
    return order[keep]


def hypervolume(F, ref):
    if len(F) == 0:
        return 0.0
    F = F[nondominated(F)]
    return float((np.diff(np.append(F[:, 0], ref[0])) * (ref[1] - F[:, 1])).sum())


def empty_front():
    return {"F": np.zeros((0, 2)), "mechs": []}


def offer(front, F_new, mech_fn, ref, cap=3000):
    """Add in-limits points to the front; mech_fn(i) builds the mechanism for new row i."""
    inside = np.where(np.isfinite(F_new).all(1) & (F_new[:, 0] < ref[0]) & (F_new[:, 1] < ref[1]))[0]
    if not len(inside):
        return
    F = np.concatenate([front["F"], F_new[inside]])
    idx = nondominated(F)
    if len(idx) > cap:
        idx = idx[np.linspace(0, len(idx) - 1, cap).round().astype(int)]
    n_old = len(front["F"])
    front["mechs"] = [front["mechs"][i] if i < n_old else mech_fn(inside[i - n_old]) for i in idx]
    front["F"] = F[idx]


SIZE_BINS = [4, 5, 6, 7, 8, 9, 10, 12, 14, 16, 20]


def pick_per_size(mechs, dist, n):
    """Indices of the n most accurate designs, round-robin over joint-count bins."""
    dist = np.asarray(dist)
    bins = np.searchsorted(SIZE_BINS, [len(m["x0"]) for m in mechs])
    ranked = [list(np.where(bins == b)[0][np.argsort(dist[bins == b])]) for b in range(len(SIZE_BINS) + 1)]
    chosen = []
    while len(chosen) < n and any(ranked):
        for r in ranked:
            if r and len(chosen) < n:
                chosen.append(r.pop(0))
    return chosen


def spread_along_front(F, n):
    """Indices of about n designs spread evenly along the front by distance."""
    idx = nondominated(F)
    if len(idx) <= n:
        return list(idx)
    return list(idx[np.linspace(0, len(idx) - 1, n).round().astype(int)])


# ---------------------------------------------------------------- Adam over a batch of designs
def new_run(mechs, weights, lr):
    """Optimizer state for a list of designs, padded to whole chunks."""
    mechs, weights = list(mechs), list(weights)
    n_real = len(mechs)
    pad = (-n_real) % CHUNK
    mechs += mechs[:pad]
    weights += weights[:pad]
    x0, A, types, tidx = pack(mechs)
    size = np.array([np.sqrt(((m["x0"] - m["x0"].mean(0)) ** 2).sum(-1).mean()) for m in mechs])
    n = len(mechs)
    return {"mechs": mechs, "n_real": n_real, "w": np.array(weights, dtype=np.float64), "step": 0,
            "x": x0, "A": A, "types": types, "tidx": tidx,
            "real": np.arange(MAX_N)[None, :] < np.array([len(m["x0"]) for m in mechs])[:, None],
            "adam_m": np.zeros_like(x0), "adam_v": np.zeros_like(x0), "lr": lr * size,
            "last_good": x0.copy(), "last_F": np.full((n, 2), np.inf)}


def adam_update(r, sl, d, m, g, step, decay):
    """Apply one Adam step to chunk `sl`; broken members go back to their last good layout.

    Returns the chunk's (distance, material) rows, inf where the member broke.
    """
    d, m, g = np.array(d), np.array(m), np.array(g, dtype=np.float64)
    g[~r["real"][sl]] = 0.0                              # padding joints carry nan gradients
    ok = np.isfinite(d) & np.isfinite(g).all(axis=(1, 2))
    F = np.full((len(d), 2), np.inf)
    F[ok] = np.stack([d[ok], m[ok]], 1)
    bad = ~ok
    r["x"][sl][bad] = r["last_good"][sl][bad]
    r["lr"][sl][bad] *= 0.5
    r["adam_m"][sl][bad] = 0.0
    r["last_good"][sl][ok] = r["x"][sl][ok]
    r["last_F"][sl][ok] = F[ok]
    g[bad] = 0.0
    mm = B1 * r["adam_m"][sl] + (1 - B1) * g
    vv = B2 * r["adam_v"][sl] + (1 - B2) * g * g
    r["adam_m"][sl], r["adam_v"][sl] = mm, vv
    upd = (mm / (1 - B1 ** step)) / (np.sqrt(vv / (1 - B2 ** step)) + EPS)
    upd[bad] = 0.0
    r["x"][sl] = r["x"][sl] - (r["lr"][sl] * decay)[:, None, None] * upd
    return F
