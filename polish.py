"""Polish the designs already on the front with small, strictly-improving steps.

reshape.py showed that restarting Adam on finished designs usually leaves them
worse than they began: its first steps are too large for the narrow minima
these designs sit in. This script never accepts a worse design. Each design
from the current best front is polished twice:

  accuracy   lower the distance without using more material than it started with
  material   lower the material without becoming less accurate than it started

A step is tried along the (projected) gradient; if it improves and respects the
cap it is kept and the step grows, otherwise it is undone and the step shrinks.
Every accepted design dominates the one before it, so the front can only move
toward the origin.

    sbatch -p mit_preemptable,mit_normal -J polish --array=0-2 -c 8 --mem=16G -t 04:00:00 \
        -o logs/%x-%A_%a.out slurm/run.sbatch polish.py
"""
import argparse
import os
import signal
import time

import numpy as np
import jax
import jax.numpy as jnp

from LINKS.CP import REFERENCE_POINTS
from LINKS.Optimization._utils import material, unscaled_distance
from slurm import ckpt
import linkcore as lc
import publish_best

p = argparse.ArgumentParser()
p.add_argument("--target", type=int, default=int(os.environ.get("SLURM_ARRAY_TASK_ID", 0)))
p.add_argument("--n", type=int, default=512, help="designs taken from along the front")
p.add_argument("--steps", type=int, default=600)
p.add_argument("--step0", type=float, default=0.002, help="first step as a fraction of mechanism size")
p.add_argument("--out", default="checkpoints/polish")
p.add_argument("--publish-every", type=float, default=1200)
args = p.parse_args()

TGT = args.target
REF = REFERENCE_POINTS[TGT]
CKPT = f"{args.out}/target_{TGT}.pkl"
os.makedirs(args.out, exist_ok=True)
target = np.load("kangaroo_target_curves.npy")[TGT].astype(np.float64)
T, THETAS = lc.T, lc.THETAS

stop = {"flag": False}
signal.signal(signal.SIGTERM, lambda *_: stop.update(flag=True))


@jax.jit
def two_grads(x0s, As, types, tidx):
    """Official distance and material, each with its own gradient."""
    tgt = jnp.broadcast_to(target, (x0s.shape[0], T, 2))

    def fd(x):
        _, d = unscaled_distance(x, As, types, THETAS, tidx, tgt)
        return jnp.where(jnp.isfinite(d), d, 0.0).sum(), d

    def fm(x):
        _, m = material(x, As)
        return m.sum(), m

    (_, d), gd = jax.value_and_grad(fd, has_aux=True)(x0s)
    (_, m), gm = jax.value_and_grad(fm, has_aux=True)(x0s)
    return d, m, gd, gm


def dot(a, b):
    return (a * b).sum(axis=(1, 2))


state = ckpt.load(CKPT)
if state is None:
    pool = np.load(publish_best.NPY, allow_pickle=True).item()[f"Problem {TGT + 1}"]
    pool = [m for m in pool if len(m["x0"]) >= 3]
    d, mat = lc.batched(lc.score, pool, 2, target)
    F = np.stack([d, mat], 1)
    chosen = lc.spread_along_front(F, args.n)
    mechs = [pool[i] for i in chosen] * 2                  # first copy: accuracy mode, second: material mode
    n_real = len(mechs)
    mechs += mechs[: (-n_real) % lc.CHUNK]
    x0, A, types, tidx = lc.pack(mechs)
    n = len(mechs)
    mode = (np.arange(n) % n_real >= n_real // 2).astype(int)
    size = np.array([np.sqrt(((m["x0"] - m["x0"].mean(0)) ** 2).sum(-1).mean()) for m in mechs])
    state = {"target": TGT, "step": 0, "elapsed": 0.0, "mechs": mechs, "n_real": n_real, "mode": mode,
             "A": A, "types": types, "tidx": tidx,
             "real": np.arange(lc.MAX_N)[None, :] < np.array([len(m["x0"]) for m in mechs])[:, None],
             "x": x0.copy(), "trial": x0.copy(), "alpha": args.step0 * size,
             "F": np.full((n, 2), np.inf), "F0": np.full((n, 2), np.inf),
             "gd": np.zeros_like(x0), "gm": np.zeros_like(x0), "accepted": np.zeros(n, dtype=int),
             "front": lc.empty_front(), "history": []}
    lc.offer(state["front"], F, lambda i: pool[i], REF)
    print(f"target {TGT}: fresh start, polishing {len(chosen)} front designs in two modes; "
          f"front HV {lc.hypervolume(F, REF):.4f}", flush=True)
else:
    print(f"target {TGT}: RESUMED at step {state['step']}, {state['elapsed'] / 3600:.2f} h used", flush=True)

s = state
N = len(s["x"])
save_due, publish_due = ckpt.Every(60), ckpt.Every(args.publish_every)


def log():
    F = s["front"]["F"]
    real = np.arange(N) < s["n_real"]
    acc, mat = real & (s["mode"] == 0), real & (s["mode"] == 1)
    gain_d = np.nanmedian(1 - s["F"][acc, 0] / s["F0"][acc, 0])
    gain_m = np.nanmedian(1 - s["F"][mat, 1] / s["F0"][mat, 1])
    s["history"].append((s["step"], s["elapsed"], lc.hypervolume(F, REF), float(gain_d), float(gain_m)))
    print(f"[{s['elapsed'] / 3600:5.2f} h] step {s['step']:4d}/{args.steps}  front HV {lc.hypervolume(F, REF):.4f} "
          f"({len(F)} designs), best distance {F[:, 0].min():.4f}  |  median distance cut {gain_d:.2%} (accuracy mode), "
          f"median material cut {gain_m:.2%} (material mode), accepted steps per design {s['accepted'][real].mean():.1f}",
          flush=True)


def publish():
    publish_best.merge_and_publish({f"Problem {TGT + 1}": s["front"]["mechs"]},
                                   f"polish target {TGT + 1}, step {s['step']}")


while s["step"] < args.steps and not stop["flag"]:
    t0 = time.monotonic()
    first = s["step"] == 0
    new_rows, new_idx = [], []
    for c in range(0, N, lc.CHUNK):
        sl = slice(c, c + lc.CHUNK)
        d, m, gd, gm = (np.array(v, dtype=np.float64) for v in two_grads(s["trial"][sl], s["A"][sl], s["types"][sl], s["tidx"][sl]))
        gd[~s["real"][sl]] = 0.0
        gm[~s["real"][sl]] = 0.0
        finite = np.isfinite(d) & np.isfinite(m) & np.isfinite(gd).all(axis=(1, 2)) & np.isfinite(gm).all(axis=(1, 2))
        Fa, F0, mode = s["F"][sl], s["F0"][sl], s["mode"][sl]
        if first:
            ok = finite
            s["F0"][sl][ok] = np.stack([d[ok], m[ok]], 1)
        else:
            better_d = (d < Fa[:, 0] - 1e-7) & (m <= F0[:, 1] * (1 + 1e-6))      # more accurate, no more material than at the start
            better_m = (m < Fa[:, 1] - 1e-7) & (d <= F0[:, 0] * (1 + 1e-6))      # lighter, no less accurate than at the start
            ok = finite & np.where(mode == 0, better_d, better_m)
        s["x"][sl][ok] = s["trial"][sl][ok]
        s["F"][sl][ok] = np.stack([d[ok], m[ok]], 1)
        s["gd"][sl][ok], s["gm"][sl][ok] = gd[ok], gm[ok]
        s["alpha"][sl] *= np.where(ok, 1.25, 0.4)
        if not first:
            s["accepted"][sl] += ok
            new_rows.append(np.stack([d[ok], m[ok]], 1))
            new_idx.append(np.where(ok)[0] + c)

        # next trial: down the gradient of the mode's objective, without climbing the other one past its cap
        g_obj = np.where((mode == 0)[:, None, None], s["gd"][sl], s["gm"][sl])
        g_cap = np.where((mode == 0)[:, None, None], s["gm"][sl], s["gd"][sl])
        at_cap = np.where(mode == 0, s["F"][sl][:, 1] >= 0.999 * s["F0"][sl][:, 1], s["F"][sl][:, 0] >= 0.999 * s["F0"][sl][:, 0])
        step_dir = -g_obj
        climb = dot(step_dir, g_cap)
        fix = at_cap & (climb > 0)
        step_dir = step_dir - np.where(fix, climb / np.maximum(dot(g_cap, g_cap), 1e-30), 0.0)[:, None, None] * g_cap
        norm = np.sqrt(np.maximum(dot(step_dir, step_dir), 1e-30))
        s["trial"][sl] = s["x"][sl] + (s["alpha"][sl] / norm)[:, None, None] * step_dir
    if new_rows and sum(len(r) for r in new_rows):
        rows, idx = np.concatenate(new_rows), np.concatenate(new_idx)
        lc.offer(s["front"], rows, lambda i: lc.with_x(s["mechs"][idx[i]], s["x"][idx[i]]), REF)
    s["step"] += 1
    s["elapsed"] += time.monotonic() - t0
    if save_due() or s["step"] == 1:
        log()
        ckpt.save(CKPT, s)
    if publish_due():
        publish()

log()
ckpt.save(CKPT, s)
publish()
print("STOPPED on signal (will resume)" if stop["flag"] else "DONE", flush=True)
