"""Wide refinement with successive halving (one target per array task).

Round one refined 136 seeds per target and its best results came from a few
seeds that happened to sit in good basins. This round starts from thousands of
candidates and spends steps where they pay off:

  stage 0  every candidate, distance only, a short run
  stage 1  the best quarter (chosen per joint-count bin), a longer run
  stage 2  the best of those, each at several material weights, a long run,
           plus the small designs from the cheap end of the fronts at
           heavier material weights

Candidates: all screening seeds, the screening fronts, and round one's final
designs. Every in-limits step is offered to the front, and the front is
published as it goes.

    sbatch -J wide --array=0-2 -c 8 --mem=32G -t 12:00:00 \
        -o logs/%x-%A_%a.out slurm/run.sbatch refine_wide.py
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

MAX_N, T, CHUNK = 20, 200, 256
SIZE_BINS = [4, 5, 6, 7, 8, 9, 10, 12, 14, 16, 20]
ACC_WEIGHTS = [0.0, 0.02, 0.05, 0.12, 0.3]        # for the accurate finalists
CHEAP_WEIGHTS = [0.05, 0.12, 0.3, 0.6]            # for the small, low-material designs

p = argparse.ArgumentParser()
p.add_argument("--target", type=int, default=int(os.environ.get("SLURM_ARRAY_TASK_ID", 0)))
p.add_argument("--n0", type=int, default=4096)
p.add_argument("--n1", type=int, default=1024)
p.add_argument("--n2", type=int, default=192)
p.add_argument("--n-cheap", type=int, default=64)
p.add_argument("--steps", type=int, nargs=3, default=[120, 300, 1200])
p.add_argument("--lr", type=float, default=0.004)
p.add_argument("--out", default="checkpoints/wide")
p.add_argument("--publish-every", type=float, default=1200)
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
        return (jnp.where(jnp.isfinite(d), d, 0.0) + w * m).sum(), (d, m)
    (_, (d, m)), g = jax.value_and_grad(loss, has_aux=True)(x0s)
    return d, m, g


def with_x(m, x):
    return {**m, "x0": np.array(x[: len(m["x0"])], dtype=np.float64)}


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


def pick_per_size(mechs, dist, n):
    """Indices of the n most accurate designs, taken round-robin over joint-count bins."""
    bins = np.searchsorted(SIZE_BINS, [len(m["x0"]) for m in mechs])
    ranked = [list(np.where(bins == b)[0][np.argsort(dist[bins == b])]) for b in range(len(SIZE_BINS) + 1)]
    chosen = []
    while len(chosen) < n and any(ranked):
        for r in ranked:
            if r and len(chosen) < n:
                chosen.append(r.pop(0))
    return chosen


def start_stage(mechs, weights):
    """Fresh optimizer state for a list of designs (padded to whole chunks)."""
    mechs, weights = list(mechs), list(weights)
    pad = (-len(mechs)) % CHUNK
    mechs += mechs[:pad]
    weights += weights[:pad]
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
    size = np.array([np.sqrt(((m["x0"] - m["x0"].mean(0)) ** 2).sum(-1).mean()) for m in mechs])
    return {"mechs": mechs, "w": np.array(weights, dtype=np.float64), "step": 0,
            "x": x0, "A": A, "types": types, "tidx": tidx,
            "real": np.arange(MAX_N)[None, :] < np.array([len(m["x0"]) for m in mechs])[:, None],
            "adam_m": np.zeros_like(x0), "adam_v": np.zeros_like(x0), "lr": args.lr * size,
            "last_good": x0.copy(), "last_F": np.full((n, 2), np.inf)}


def gather_candidates():
    """Screening seeds and fronts, plus round one's final designs."""
    seeds, seed_d, cheap = [], [], []
    for f in sorted(glob.glob("checkpoints/screen/worker_*.pkl")):
        s = ckpt.load(f)
        if not s:
            continue
        seeds += s["seeds"][TGT]["mechs"]
        seed_d += list(s["seeds"][TGT]["F"][:, 0])
        cheap += [(F[1], m) for F, m in zip(s["fronts"][TGT]["F"], s["fronts"][TGT]["mechs"])]
    r1 = ckpt.load(f"checkpoints/refine/target_{TGT}.pkl")
    first = [with_x(m, x) for m, x in zip(r1["mechs"], r1["last_good"])] if r1 else []
    if r1:                                               # round one's cheap end, too
        for F, x, src in zip(r1["front"]["F"], r1["front"]["x"], r1["front"]["src"]):
            cheap.append((F[1], with_x(r1["mechs"][src], x)))
    ok = [i for i, m in enumerate(seeds) if len(m["x0"]) >= 4]
    seeds, seed_d = [seeds[i] for i in ok], np.array(seed_d)[ok]
    take = pick_per_size(seeds, seed_d, max(0, args.n0 - len(first)))
    cheap = [m for _, m in sorted(cheap, key=lambda c: c[0]) if 4 <= len(m["x0"]) <= 6][: args.n_cheap]
    return first + [seeds[i] for i in take], cheap


# ---------------------------------------------------------------- state
state = ckpt.load(CKPT)
if state is None:
    cands, cheap = gather_candidates()
    state = {"target": TGT, "stage": 0, "elapsed": 0.0, "cheap": cheap,
             "run": start_stage(cands, [0.0] * len(cands)),
             "front": {"F": np.zeros((0, 2)), "mechs": []}, "history": []}
    sizes = np.bincount([len(m["x0"]) for m in cands], minlength=MAX_N + 1)
    print(f"target {TGT}: fresh start, {len(cands)} candidates, joint counts "
          f"{ {k: int(v) for k, v in enumerate(sizes) if v} }, {len(cheap)} cheap designs held for stage 2", flush=True)
else:
    print(f"target {TGT}: RESUMED in stage {state['stage']} at step {state['run']['step']}, "
          f"{state['elapsed'] / 3600:.2f} h used", flush=True)

B1, B2, EPS = 0.9, 0.999, 1e-8
save_due, publish_due = ckpt.Every(60), ckpt.Every(args.publish_every)


def offer(F_new, mech_fn):
    fr = state["front"]
    inside = np.where(np.isfinite(F_new).all(1) & (F_new[:, 0] < REF[0]) & (F_new[:, 1] < REF[1]))[0]
    if not len(inside):
        return
    F = np.concatenate([fr["F"], F_new[inside]])
    idx = nondominated(F)
    if len(idx) > 3000:
        idx = idx[np.linspace(0, len(idx) - 1, 3000).round().astype(int)]
    n_old = len(fr["F"])
    fr["mechs"] = [fr["mechs"][i] if i < n_old else mech_fn(inside[i - n_old]) for i in idx]
    fr["F"] = F[idx]


def log():
    F, r = state["front"]["F"], state["run"]
    d = r["last_F"][:, 0]
    best = F[:, 0].min() if len(F) else float("nan")
    state["history"].append((state["stage"], r["step"], state["elapsed"], hypervolume(F), float(best)))
    print(f"[{state['elapsed'] / 3600:5.2f} h] stage {state['stage']} step {r['step']:4d}/{args.steps[state['stage']]}  "
          f"members {len(d)}  front HV {hypervolume(F):.3f} ({len(F)} designs)  best distance {best:.4f}  "
          f"median {np.median(d[np.isfinite(d)]) if np.isfinite(d).any() else float('nan'):.3f}", flush=True)


def publish():
    publish_best.merge_and_publish({f"Problem {TGT + 1}": state["front"]["mechs"]},
                                   f"wide refine target {TGT + 1}, stage {state['stage']} step {state['run']['step']}")


while state["stage"] < 3 and not stop["flag"]:
    r = state["run"]
    N, n_steps = len(r["x"]), args.steps[state["stage"]]
    while r["step"] < n_steps and not stop["flag"]:
        t0 = time.monotonic()
        step = r["step"] + 1
        decay = 0.3 ** (r["step"] / n_steps)
        F_step = np.full((N, 2), np.inf)
        x_before = r["x"].copy()
        for c in range(0, N, CHUNK):
            sl = slice(c, c + CHUNK)
            d, m, g = value_and_grad(r["x"][sl], r["A"][sl], r["types"][sl], r["tidx"][sl], r["w"][sl])
            d, m, g = np.array(d), np.array(m), np.array(g, dtype=np.float64)
            g[~r["real"][sl]] = 0.0
            ok = np.isfinite(d) & np.isfinite(g).all(axis=(1, 2))
            F_step[sl][ok] = np.stack([d[ok], m[ok]], 1)
            bad = ~ok
            r["x"][sl][bad] = r["last_good"][sl][bad]
            r["lr"][sl][bad] *= 0.5
            r["adam_m"][sl][bad] = 0.0
            r["last_good"][sl][ok] = r["x"][sl][ok]
            r["last_F"][sl][ok] = F_step[sl][ok]
            g[bad] = 0.0
            mm = B1 * r["adam_m"][sl] + (1 - B1) * g
            vv = B2 * r["adam_v"][sl] + (1 - B2) * g * g
            r["adam_m"][sl], r["adam_v"][sl] = mm, vv
            upd = (mm / (1 - B1 ** step)) / (np.sqrt(vv / (1 - B2 ** step)) + EPS)
            upd[bad] = 0.0
            r["x"][sl] = r["x"][sl] - (r["lr"][sl] * decay)[:, None, None] * upd
        offer(F_step, lambda i: with_x(r["mechs"][i], x_before[i]))
        r["step"] = step
        state["elapsed"] += time.monotonic() - t0
        if save_due() or step == 1:
            log()
            ckpt.save(CKPT, state)
        if publish_due():
            publish()
    if r["step"] < n_steps:
        break                                             # stopped by signal

    # ---- stage finished: keep the best and move on
    log()
    finals = [with_x(m, x) for m, x in zip(r["mechs"], r["last_good"])]
    dist = r["last_F"][:, 0]
    if state["stage"] == 0:
        keep = pick_per_size(finals, dist, args.n1)
        state["run"] = start_stage([finals[i] for i in keep], [0.0] * len(keep))
    elif state["stage"] == 1:
        keep = pick_per_size(finals, dist, args.n2)
        mechs = [finals[i] for i in keep for _ in ACC_WEIGHTS] + [m for m in state["cheap"] for _ in CHEAP_WEIGHTS]
        weights = ACC_WEIGHTS * len(keep) + CHEAP_WEIGHTS * len(state["cheap"])
        state["run"] = start_stage(mechs, weights)
    state["stage"] += 1
    ckpt.save(CKPT, state)
    publish()

ckpt.save(CKPT, state)
print("STOPPED on signal (will resume)" if stop["flag"] else "DONE", flush=True)
