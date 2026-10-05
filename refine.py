"""Gradient refinement of the screening seeds for one target.

Takes the most accurate designs the screen found (a fixed number per joint
count, so larger mechanisms get their chance too) and runs Adam on the joint
positions. Each seed is refined several times with a different material weight
w, minimizing  distance + w * material, so one seed spreads into several
points along the accuracy/material trade-off. Every step's (distance,
material) is the official metric, so every in-limits iterate is offered to the
Pareto front.

Preempt-safe: positions, Adam state and the front are checkpointed every
minute; the seed selection is stored in the checkpoint so a resume is exact.
One array task per target:

    sbatch -J refine --array=0-2 -c 8 --mem=24G -t 08:00:00 \
        -o logs/%x-%A_%a.out slurm/run.sbatch refine.py
"""
import argparse
import glob
import os
import signal
import time

os.environ["JAX_PLATFORMS"] = "cpu"

import numpy as np
import jax
import jax.numpy as jnp

from LINKS.CP import REFERENCE_POINTS
from LINKS.Optimization._utils import material, unscaled_distance
from slurm import ckpt
import publish_best

MAX_N, T = 20, 200
CHUNK = 256                                   # fixed batch for the jitted gradient
WEIGHTS = [0.0, 0.03, 0.1, 0.3]               # material weights, one refinement per seed each
PER_SIZE = {4: 24, 5: 24, 6: 32, 7: 32, 8: 24, 9: 24}   # seeds per joint count (9 = nine or more)

p = argparse.ArgumentParser()
p.add_argument("--target", type=int, default=int(os.environ.get("SLURM_ARRAY_TASK_ID", 0)))
p.add_argument("--steps", type=int, default=1200)
p.add_argument("--lr", type=float, default=0.004, help="Adam step as a fraction of mechanism size")
p.add_argument("--seed-dir", default="checkpoints/screen")
p.add_argument("--out", default="checkpoints/refine")
p.add_argument("--publish-every", type=float, default=1200, help="seconds between publishes")
args = p.parse_args()

TGT = args.target
REF = REFERENCE_POINTS[TGT]
CKPT = f"{args.out}/target_{TGT}.pkl"
os.makedirs(args.out, exist_ok=True)
target = np.load("kangaroo_target_curves.npy")[TGT]
THETAS = np.linspace(0.0, 2 * np.pi, T)

stop = {"flag": False}
signal.signal(signal.SIGTERM, lambda *_: stop.update(flag=True))


@jax.jit
def value_and_grad(x0s, As, types, tidx, w):
    def loss(x):
        _, d = unscaled_distance(x, As, types, THETAS, tidx, jnp.broadcast_to(target, (x.shape[0], T, 2)))
        _, m = material(x, As)
        safe = jnp.where(jnp.isfinite(d), d, 0.0)
        return (safe + w * m).sum(), (d, m)
    (_, (d, m)), g = jax.value_and_grad(loss, has_aux=True)(x0s)
    return d, m, g


def pack(mechs):
    """Padded solver arrays. Screen seeds are already in solver order with the traced joint last."""
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


def nondominated(F):
    order = np.lexsort((F[:, 1], F[:, 0]))
    best_m = np.minimum.accumulate(F[order, 1])
    keep = np.ones(len(F), dtype=bool)
    keep[1:] = F[order, 1][1:] < best_m[:-1]
    return order[keep]


def hypervolume(F):
    if len(F) == 0:
        return 0.0
    F = F[nondominated(F)]
    return float((np.diff(np.append(F[:, 0], REF[0])) * (REF[1] - F[:, 1])).sum())


def select_seeds():
    pool_F, pool_m = [], []
    for f in sorted(glob.glob(f"{args.seed_dir}/worker_*.pkl")):
        s = ckpt.load(f)
        if s:
            pool_F.append(s["seeds"][TGT]["F"])
            pool_m += s["seeds"][TGT]["mechs"]
    F = np.concatenate(pool_F)
    size = np.minimum(np.array([len(m["x0"]) for m in pool_m]), 9)
    chosen = []
    for n, k in PER_SIZE.items():
        idx = np.where(size == n)[0]
        chosen += list(idx[np.argsort(F[idx, 0])[:k]])
    return [pool_m[i] for i in chosen], F[chosen]


# ---------------------------------------------------------------- state
state = ckpt.load(CKPT)
if state is None:
    seeds, seedF = select_seeds()
    mechs = [s for s in seeds for _ in WEIGHTS]
    w = np.tile(np.array(WEIGHTS), len(seeds))
    pad = (-len(mechs)) % CHUNK                      # fill the last chunk with repeats
    mechs += mechs[:pad]
    w = np.concatenate([w, w[:pad]])
    x0, A, types, tidx = pack(mechs)
    size = np.array([np.sqrt(((m["x0"] - m["x0"].mean(0)) ** 2).sum(-1).mean()) for m in mechs])  # rms radius
    state = {"target": TGT, "step": 0, "elapsed": 0.0, "mechs": mechs, "w": w,
             "x": x0, "A": A, "types": types, "tidx": tidx,
             "real": np.arange(MAX_N)[None, :] < np.array([len(m["x0"]) for m in mechs])[:, None],
             "adam_m": np.zeros_like(x0), "adam_v": np.zeros_like(x0),
             "lr": args.lr * size, "last_good": x0.copy(),
             "front": {"F": np.zeros((0, 2)), "x": np.zeros((0, MAX_N, 2)), "src": np.zeros(0, dtype=int)},
             "seed_F": seedF, "history": []}
    print(f"target {TGT}: fresh start, {len(seeds)} seeds x {len(WEIGHTS)} weights = {len(mechs)} members; "
          f"seed sizes {np.bincount([len(s['x0']) for s in seeds])[4:].tolist()} (4 joints up); "
          f"best seed distance {seedF[:, 0].min():.3f}", flush=True)
else:
    print(f"target {TGT}: RESUMED at step {state['step']}, {state['elapsed'] / 3600:.2f} h used", flush=True)

N = len(state["x"])
B1, B2, EPS = 0.9, 0.999, 1e-8


def front_mechs():
    """Front as submission dicts (structure from the member, positions from the front)."""
    out = []
    for xi, src in zip(state["front"]["x"], state["front"]["src"]):
        m = state["mechs"][src]
        out.append({"x0": xi[: len(m["x0"])].copy(), "edges": m["edges"], "fixed_joints": m["fixed_joints"],
                    "motor": m["motor"], "target_joint": m["target_joint"]})
    return out


def publish():
    publish_best.merge_and_publish({f"Problem {TGT + 1}": front_mechs()},
                                   f"refine target {TGT + 1}, step {state['step']}")


def checkpoint(d_now):
    hv = hypervolume(state["front"]["F"])
    best = float(state["front"]["F"][:, 0].min()) if len(state["front"]["F"]) else float("nan")
    state["history"].append((state["step"], state["elapsed"], hv, best))
    ckpt.save(CKPT, state)
    print(f"[{state['elapsed'] / 3600:5.2f} h] step {state['step']:5d}  HV {hv:.3f}  front {len(state['front']['F']):4d}  "
          f"best distance {best:.4f}  working members {int(np.isfinite(d_now).sum())}/{len(d_now)}  "
          f"median distance {np.median(d_now[np.isfinite(d_now)]) if np.isfinite(d_now).any() else float('nan'):.3f}",
          flush=True)


save_due, publish_due = ckpt.Every(60), ckpt.Every(args.publish_every)
d_all = np.full(N, np.nan)
while state["step"] < args.steps and not stop["flag"]:
    t0 = time.monotonic()
    step = state["step"] + 1
    decay = 0.3 ** (state["step"] / args.steps)          # step size decays to 30%
    newF, newx, newsrc = [], [], []
    for c in range(0, N, CHUNK):
        sl = slice(c, c + CHUNK)
        d, m, g = value_and_grad(state["x"][sl], state["A"][sl], state["types"][sl], state["tidx"][sl], state["w"][sl])
        d, m, g = np.array(d), np.array(m), np.array(g, dtype=np.float64)
        g[~state["real"][sl]] = 0.0          # padding joints carry meaningless (nan) gradients
        ok = np.isfinite(d) & np.isfinite(g).all(axis=(1, 2))
        d_all[sl] = np.where(ok, d, np.inf)

        inside = np.where(ok & (d < REF[0]) & (m < REF[1]))[0]
        if len(inside):
            newF.append(np.stack([d[inside], m[inside]], 1))
            newx.append(state["x"][sl][inside].copy())
            newsrc.append(inside + c)

        # members that broke (locked up) go back to their last good layout with a smaller step
        bad = ~ok
        state["x"][sl][bad] = state["last_good"][sl][bad]
        state["lr"][sl][bad] *= 0.5
        state["adam_m"][sl][bad] = 0.0
        state["last_good"][sl][ok] = state["x"][sl][ok]

        g[bad] = 0.0
        mm = B1 * state["adam_m"][sl] + (1 - B1) * g
        vv = B2 * state["adam_v"][sl] + (1 - B2) * g * g
        state["adam_m"][sl], state["adam_v"][sl] = mm, vv
        upd = (mm / (1 - B1 ** step)) / (np.sqrt(vv / (1 - B2 ** step)) + EPS)
        upd[bad] = 0.0
        state["x"][sl] = state["x"][sl] - (state["lr"][sl] * decay)[:, None, None] * upd

    if newF:
        fr = state["front"]
        F = np.concatenate([fr["F"]] + newF)
        X = np.concatenate([fr["x"]] + newx)
        S = np.concatenate([fr["src"]] + newsrc)
        idx = nondominated(F)
        if len(idx) > 3000:
            idx = idx[np.linspace(0, len(idx) - 1, 3000).round().astype(int)]
        fr["F"], fr["x"], fr["src"] = F[idx], X[idx], S[idx]

    state["step"] = step
    state["elapsed"] += time.monotonic() - t0
    if save_due() or step == 1:
        checkpoint(d_all)
    if publish_due():
        publish()

checkpoint(d_all)
publish()
print("STOPPED on signal (will resume)" if stop["flag"] else "DONE", flush=True)
