"""Refine with a more forgiving objective first, then polish on the official one.

The official distance pairs points by position along the curve, so a design
that grows half a tail gets almost no credit and plain gradient descent never
starts one. This script runs the same parents through three arms and compares
them, always measuring with the official distance:

  control  official distance all the way (what refine.py does)
  chamfer  starts on a nearest-point (two-way chamfer) distance, which pays for
           partial features, blends back, and finishes on the official distance
  smooth   official distance against a blurred target (few Fourier harmonics)
           that is sharpened in steps until it is the real target

Parents are taken from all along the current best front, not just its tip.
Every in-limits step of every arm is offered to the front (scored officially).

    sbatch -p mit_preemptable,mit_normal -J reshape --array=0-2 -c 8 --mem=16G -t 04:00:00 \
        -o logs/%x-%A_%a.out slurm/run.sbatch reshape.py
"""
import argparse
import os
import signal
import time

import numpy as np
import jax
import jax.numpy as jnp

from LINKS.CP import REFERENCE_POINTS
from LINKS.Geometry._utils import equisample, find_optimal_correspondences
from LINKS.Kinematics._solvers import dyadic_solve
from LINKS.Optimization._utils import material
from slurm import ckpt
import linkcore as lc
import publish_best

ARMS = ["control", "chamfer", "smooth"]
WEIGHTS = [0.0, 0.08]
HARMONICS = [4, 6, 9, 14, 22]                # blur levels for the smooth arm, coarse to fine

p = argparse.ArgumentParser()
p.add_argument("--target", type=int, default=int(os.environ.get("SLURM_ARRAY_TASK_ID", 0)))
p.add_argument("--steps", type=int, default=1000)
p.add_argument("--n-front", type=int, default=48, help="parents spread along the front")
p.add_argument("--n-accurate", type=int, default=48, help="most accurate parents per joint-count bin")
p.add_argument("--lr", type=float, default=0.004)
p.add_argument("--out", default="checkpoints/reshape")
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


def blurred(k):
    """The target rebuilt from its lowest k Fourier harmonics."""
    t = np.array(equisample(jnp.asarray(target[None]), T))[0]
    z = np.fft.fft(t[:, 0] + 1j * t[:, 1])
    freq = np.abs(np.fft.fftfreq(T, 1 / T))
    z[freq > k] = 0
    z = np.fft.ifft(z)
    return np.stack([z.real, z.imag], -1)


TARGETS_S = np.stack([blurred(k) for k in HARMONICS] + [target])     # last level is the real target


@jax.jit
def value_and_grad(x0s, As, types, tidx, w, a, lam, tgt_s):
    """Gradient of  a * official(vs tgt_s) + lam * chamfer(vs real target) + w * material.

    Also returns the official distance against the real target, and the material.
    """
    tgt = jnp.broadcast_to(target, (x0s.shape[0], T, 2))

    def loss(x):
        sol = dyadic_solve(As, x, types, THETAS)
        bad = jnp.isnan(sol).any(axis=(1, 2, 3))
        c = sol[jnp.arange(sol.shape[0]), tidx]
        c = jnp.nan_to_num(c * (~bad)[:, None, None], nan=0.0) + tgt * bad[:, None, None]
        nc = equisample(c, T)
        nc = nc - nc.mean(1, keepdims=True)

        def norm(tg):
            nt = equisample(tg, T)
            nt = nt - nt.mean(1, keepdims=True)
            s = jnp.sqrt((nt ** 2).sum(-1).sum(-1) / T)[:, None, None]
            return nt / s, s

        nt, s = norm(tgt)
        nts, ss = norm(tgt_s)
        _, aligned, d_true = find_optimal_correspondences(nc / s, nt, return_rotation_matrices=False,
                                                          return_aligned_curves=True, return_distances=True)
        _, d_s = find_optimal_correspondences(nc / ss, nts, return_rotation_matrices=False,
                                              return_aligned_curves=False, return_distances=True)
        gaps = jnp.sqrt(((aligned[:, :, None, :] - nt[:, None, :, :]) ** 2).sum(-1) + 1e-12)   # [B, T, T]
        chamfer = (gaps.min(2).mean(1) + gaps.min(1).mean(1)) * jnp.pi      # same units as the distance
        _, m = material(x, As)
        per = a * d_s + lam * chamfer + w * m
        return jnp.where(bad, 0.0, per).sum(), (jnp.where(bad, jnp.inf, d_true), m)

    (_, (d, m)), g = jax.value_and_grad(loss, has_aux=True)(x0s)
    return d, m, g


def schedule(arm, frac):
    """(weight on official term, weight on chamfer term, blur level) for an arm at this point of the run."""
    last = len(HARMONICS)
    if arm == 1:                                          # chamfer, blend back, then official only
        if frac < 0.4:
            return 0.3, 1.0, last
        if frac < 0.7:
            u = (frac - 0.4) / 0.3
            return 0.3 + 0.7 * u, 1.0 - u, last
    if arm == 2 and frac < 0.7:                           # sharpen the blurred target in steps
        return 1.0, 0.0, int(frac / 0.7 * len(HARMONICS))
    return 1.0, 0.0, last


# ---------------------------------------------------------------- state
state = ckpt.load(CKPT)
if state is None:
    pool = np.load(publish_best.NPY, allow_pickle=True).item()[f"Problem {TGT + 1}"]
    pool = [m for m in pool if len(m["x0"]) >= 4]
    d, m = lc.batched(lc.score, pool, 2, target)
    F = np.stack([d, m], 1)
    chosen = list(dict.fromkeys(lc.spread_along_front(F, args.n_front) + lc.pick_per_size(pool, d, args.n_accurate)))
    parents = [pool[i] for i in chosen]
    mechs, weights, arm, parent = [], [], [], []
    for pi, par in enumerate(parents):
        for a in range(len(ARMS)):
            for w in WEIGHTS:
                mechs.append(par); weights.append(w); arm.append(a); parent.append(pi)
    run = lc.new_run(mechs, weights, args.lr)
    pad = len(run["mechs"]) - len(mechs)
    state = {"target": TGT, "elapsed": 0.0, "run": run,
             "arm": np.array(arm + arm[:pad]), "parent": np.array(parent + parent[:pad]),
             "parents_F": F[chosen], "best_F": np.full((len(run["mechs"]), 2), np.inf),
             "front": lc.empty_front(), "arm_fronts": [np.zeros((0, 2)) for _ in ARMS], "history": []}
    lc.offer(state["front"], F, lambda i: pool[i], REF)
    print(f"target {TGT}: fresh start, {len(parents)} parents (distance {F[chosen, 0].min():.3f} to "
          f"{F[chosen, 0].max():.3f}) x {len(ARMS)} arms x {len(WEIGHTS)} weights = {len(mechs)} members; "
          f"starting front HV {lc.hypervolume(F, REF):.3f}", flush=True)
else:
    print(f"target {TGT}: RESUMED at step {state['run']['step']}, {state['elapsed'] / 3600:.2f} h used", flush=True)

r = state["run"]
N = len(r["x"])
save_due, publish_due = ckpt.Every(60), ckpt.Every(args.publish_every)


def compare():
    """Per arm: its own front's hypervolume, and how its members did against the control arm."""
    real = np.arange(N) < r["n_real"]
    out = {}
    for a, name in enumerate(ARMS):
        sel = real & (state["arm"] == a)
        out[name] = {"front_hv": lc.hypervolume(state["arm_fronts"][a], REF),
                     "best_distance": float(state["best_F"][sel, 0].min()),
                     "median_best_distance": float(np.median(state["best_F"][sel & (r["w"] == 0), 0]))}
    ctrl = state["best_F"][real & (state["arm"] == 0), 0]
    for a in (1, 2):
        mine = state["best_F"][real & (state["arm"] == a), 0]       # same parent and weight order as control
        both = np.isfinite(ctrl) & np.isfinite(mine)
        out[ARMS[a]]["beats_control"] = f"{int((mine[both] < ctrl[both] - 1e-4).sum())}/{int(both.sum())}"
        out[ARMS[a]]["median_change_vs_control"] = float(np.median(mine[both] - ctrl[both])) if both.any() else None
    return out


def log():
    c = compare()
    F = state["front"]["F"]
    state["history"].append((r["step"], state["elapsed"], lc.hypervolume(F, REF), c))
    print(f"[{state['elapsed'] / 3600:5.2f} h] step {r['step']:4d}/{args.steps}  front HV {lc.hypervolume(F, REF):.3f} "
          f"({len(F)} designs), best distance {F[:, 0].min():.4f}  ||  " +
          "  |  ".join(f"{k}: HV {v['front_hv']:.3f}, best {v['best_distance']:.4f}, median {v['median_best_distance']:.4f}"
                       + (f", beats control {v['beats_control']}" if "beats_control" in v else "")
                       for k, v in c.items()), flush=True)


def publish():
    publish_best.merge_and_publish({f"Problem {TGT + 1}": state["front"]["mechs"]},
                                   f"reshape target {TGT + 1}, step {r['step']}")


while r["step"] < args.steps and not stop["flag"]:
    t0 = time.monotonic()
    step = r["step"] + 1
    frac = r["step"] / args.steps
    sched = [schedule(a, frac) for a in range(len(ARMS))]
    a_vec = np.array([sched[a][0] for a in state["arm"]])
    lam_vec = np.array([sched[a][1] for a in state["arm"]])
    lvl = np.array([sched[a][2] for a in state["arm"]])
    x_before = r["x"].copy()
    F_step = np.full((N, 2), np.inf)
    for c in range(0, N, lc.CHUNK):
        sl = slice(c, c + lc.CHUNK)
        d, m, g = value_and_grad(r["x"][sl], r["A"][sl], r["types"][sl], r["tidx"][sl], r["w"][sl],
                                 a_vec[sl], lam_vec[sl], TARGETS_S[lvl[sl]])
        F_step[sl] = lc.adam_update(r, sl, d, m, g, step, 0.3 ** frac)
    feasible = np.isfinite(F_step).all(1) & (F_step[:, 1] < REF[1])
    better = feasible & (F_step[:, 0] < state["best_F"][:, 0])
    state["best_F"][better] = F_step[better]
    lc.offer(state["front"], F_step, lambda i: lc.with_x(r["mechs"][i], x_before[i]), REF)
    for a in range(len(ARMS)):
        rows = F_step[(state["arm"] == a) & feasible & (F_step[:, 0] < REF[0])]
        if len(rows):
            Fa = np.concatenate([state["arm_fronts"][a], rows])
            state["arm_fronts"][a] = Fa[lc.nondominated(Fa)]
    r["step"] = step
    state["elapsed"] += time.monotonic() - t0
    if save_due() or step == 1:
        log()
        ckpt.save(CKPT, state)
    if publish_due():
        publish()

log()
ckpt.save(CKPT, state)
publish()
print("STOPPED on signal (will resume)" if stop["flag"] else "DONE", flush=True)
