"""Mass-sample random mechanisms and screen them against all three targets.

Each batch:
  1. Sample random skeletons (course MechanismRandomizer) and several random
     joint layouts per skeleton; solve them all in one batched call.
  2. Every non-fixed joint with a finite trajectory is a candidate. Its
     mechanism is pruned to the joint's ancestors (the rest only adds material).
  3. Cheap shape screen (FFT correlation = least-squares alignment over all
     cyclic shifts, both directions, rotation and scale) against each target.
  4. The best candidates per target are evaluated with the official
     `Tools` metric over a grid of scales. Scaling a mechanism scales its
     curve and its material together, so each candidate yields several
     distance/material trade-off points.
  5. Points inside the target's limits update that target's Pareto front; the
     most accurate candidates are also kept as seeds for later refinement.

Preempt-safe: state changes only at batch boundaries, batches are seeded by
(seed, batch index), and the checkpoint is written atomically every minute.
Run several workers as a Slurm array; each uses its own seed and checkpoint.

    sbatch -J screen --array=0-3 -c 8 --mem=16G -t 12:00:00 \
        -o logs/%x-%A_%a.out slurm/run.sbatch screen.py --hours 8
"""
import argparse
import json
import os
import signal
import time

os.environ["JAX_PLATFORMS"] = "cpu"

import numpy as np
import jax
import jax.numpy as jnp

from LINKS.CP import REFERENCE_POINTS
from LINKS.Geometry._utils import equisample
from LINKS.Kinematics._solvers import dyadic_solve
from LINKS.Optimization import Tools
from slurm import ckpt

MAX_N = 20          # joint limit of the challenge and of the solver padding
T = 200             # timesteps, as in evaluate_submission
N_SKEL = 128        # skeletons per batch
N_POS = 8           # random layouts per skeleton
B = N_SKEL * N_POS  # mechanisms per solve
TOP_K = 40          # candidates per target per batch sent to the official metric
SCALE_GRID = np.array([0.72, 0.80, 0.86, 0.90, 0.94, 0.97, 1.0, 1.03, 1.07])
EVAL_B = 512        # fixed batch for the official metric (avoids re-jitting)
N_SEEDS = 2000      # most accurate candidates kept per target
FRONT_CAP = 4000

p = argparse.ArgumentParser()
p.add_argument("--hours", type=float, default=8.0, help="compute budget for this worker")
p.add_argument("--seed", type=int, default=2156)
p.add_argument("--out", default="checkpoints/screen")
args = p.parse_args()

WORKER = int(os.environ.get("SLURM_ARRAY_TASK_ID", 0))
CKPT = f"{args.out}/worker_{WORKER}.pkl"
SUMMARY = f"{args.out}/worker_{WORKER}.json"
N_TARGETS = len(REFERENCE_POINTS)
os.makedirs(args.out, exist_ok=True)

stop = {"flag": False}
signal.signal(signal.SIGTERM, lambda *_: stop.update(flag=True))

# ---------------------------------------------------------------- targets
targets = np.load("kangaroo_target_curves.npy").astype(np.float64)
t_eq = np.array(equisample(jnp.asarray(targets), T))
t_eq = t_eq - t_eq.mean(1, keepdims=True)
S_TARGET = np.sqrt((t_eq ** 2).sum(-1).sum(-1) / T)            # rms radius of each target
t_unit = t_eq / S_TARGET[:, None, None]
FT_CONJ = jnp.conj(jnp.fft.fft(jnp.asarray(t_unit[..., 0] + 1j * t_unit[..., 1])))  # [3, T]
THETAS = np.linspace(0.0, 2 * np.pi, T)


@jax.jit
def solve_and_screen(As, x0s, types):
    """Solve a batch and score every joint's curve against every target.

    Returns (A, s) of shape [B, MAX_N, 3]: A is the normalized least-squares
    mismatch in [0, 1] (nan if the joint has no usable curve) and s is the
    least-squares scale, in units of the target's rms radius.
    """
    sol = dyadic_solve(As, x0s, types, THETAS)                  # [B, N, T, 2]
    curves = sol.reshape(-1, T, 2)
    finite = jnp.isfinite(curves).all(axis=(1, 2))
    safe = jnp.where(finite[:, None, None], curves, jnp.asarray(t_eq[0], dtype=curves.dtype))
    eq = equisample(safe, T)
    eq = eq - eq.mean(1, keepdims=True)
    z = eq[..., 0] + 1j * eq[..., 1]
    cc = (jnp.abs(z) ** 2).sum(-1)
    Fz = jnp.fft.fft(z)
    Fzr = jnp.fft.fft(z[:, ::-1])
    r1 = jnp.abs(jnp.fft.ifft(Fz[:, None, :] * FT_CONJ[None])).max(-1)
    r2 = jnp.abs(jnp.fft.ifft(Fzr[:, None, :] * FT_CONJ[None])).max(-1)
    peak = jnp.maximum(r1, r2)                                   # [B*N, 3]
    A = 1.0 - peak ** 2 / (cc[:, None] * T)
    s = peak / cc[:, None]
    ok = finite & (cc > 1e-10)
    A = jnp.where(ok[:, None], A, jnp.nan)
    return A.reshape(-1, MAX_N, N_TARGETS), s.reshape(-1, MAX_N, N_TARGETS)


# ---------------------------------------------------------------- skeletons
def make_skeleton():
    """One random dyadic skeleton in solver order, with pruning tables.

    Same construction as the course MechanismRandomizer (ground 0, crank 1,
    each further joint is either ground or hangs off two earlier joints, not
    both ground) but without its every-joint-must-be-used rule: unused joints
    are pruned away per candidate anyway, and that rule recurses without bound
    at higher ground probabilities.
    """
    n = np.random.randint(10, MAX_N + 1)
    p_fixed = np.random.uniform(0.1, 0.35)
    fixed_flag = np.zeros(n, dtype=bool)
    fixed_flag[[0, 2]] = True
    fixed_flag[4:] = np.random.random(n - 4) < p_fixed
    raw_edges = [(0, 1)]
    for i in range(3, n):
        if fixed_flag[i]:
            continue
        a, b = np.random.choice(i, size=2, replace=False)
        while fixed_flag[a] and fixed_flag[b]:
            a, b = np.random.choice(i, size=2, replace=False)
        raw_edges += [(a, i), (b, i)]
    # solver order: motor ground, crank, the other grounds, then moving joints
    order = np.concatenate([[0, 1], np.where(fixed_flag)[0][1:], np.where(~fixed_flag)[0][1:]])
    label = np.argsort(order)
    edges = label[np.array(raw_edges)]
    fixed = label[np.where(fixed_flag)[0]]
    par, child = edges.min(1), edges.max(1)
    is_fixed = np.zeros(MAX_N, dtype=bool)
    is_fixed[fixed] = True
    anc = np.zeros((MAX_N, MAX_N), dtype=bool)       # anc[k] = nodes needed to drive k
    for k in range(n):
        anc[k, k] = True
        if not is_fixed[k]:
            for q in par[child == k]:
                anc[k] |= anc[q]
    needs = anc[:, child]                            # [node, edge]: edge kept when pruning to node
    return {"n": n, "par": par, "child": child, "is_fixed": is_fixed, "anc": anc, "needs": needs}


def pruned(skel, x0, k, scale):
    """Sub-mechanism that drives joint k (which becomes the last joint)."""
    keep = np.where(skel["anc"][k])[0]
    remap = -np.ones(MAX_N, dtype=int)
    remap[keep] = np.arange(len(keep))
    e = skel["needs"][k]
    edges = np.stack([remap[skel["par"][e]], remap[skel["child"][e]]], 1)
    return {
        "x0": x0[keep] * scale,
        "edges": edges,
        "fixed_joints": remap[np.where(skel["is_fixed"][: skel["n"]] & skel["anc"][k][: skel["n"]])[0]],
        "motor": np.array([0, 1]),
        "target_joint": len(keep) - 1,
    }


# ---------------------------------------------------------------- official metric
tools = Tools(timesteps=T, max_size=MAX_N, material=True, scaled=False, device="cpu")
tools.compile()


def official(mechs, target):
    """Distance and material from the scoring code, in fixed-size batches."""
    d_all, m_all = [], []
    for i in range(0, len(mechs), EVAL_B):
        chunk = mechs[i:i + EVAL_B]
        pad = chunk + [chunk[0]] * (EVAL_B - len(chunk))
        d, m = tools([c["x0"] for c in pad], [c["edges"] for c in pad],
                     [c["fixed_joints"] for c in pad], [c["motor"] for c in pad],
                     target, [c["target_joint"] for c in pad])
        d_all.append(np.asarray(d)[: len(chunk)])
        m_all.append(np.asarray(m)[: len(chunk)])
    return np.concatenate(d_all), np.concatenate(m_all)


# ---------------------------------------------------------------- archives
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
    widths = np.diff(np.append(F[:, 0], ref[0]))
    return float((widths * (ref[1] - F[:, 1])).sum())


def update_front(front, F_new, mechs_new):
    F = np.concatenate([front["F"], F_new])
    mechs = front["mechs"] + mechs_new
    idx = nondominated(F)
    if len(idx) > FRONT_CAP:
        idx = idx[np.linspace(0, len(idx) - 1, FRONT_CAP).round().astype(int)]
    front["F"], front["mechs"] = F[idx], [mechs[i] for i in idx]


def update_seeds(seeds, F_new, mechs_new):
    F = np.concatenate([seeds["F"], F_new])
    mechs = seeds["mechs"] + mechs_new
    idx = np.argsort(F[:, 0])[:N_SEEDS]
    seeds["F"], seeds["mechs"] = F[idx], [mechs[i] for i in idx]


def empty():
    return {"F": np.zeros((0, 2)), "mechs": []}


# ---------------------------------------------------------------- state
state = ckpt.load(CKPT)
if state is None:
    state = {"worker": WORKER, "seed": args.seed, "batch": 0, "elapsed": 0.0,
             "n_mech": 0, "n_curves": 0, "n_official": 0,
             "fronts": [empty() for _ in range(N_TARGETS)],
             "seeds": [empty() for _ in range(N_TARGETS)],
             "history": []}
    print(f"worker {WORKER}: fresh start", flush=True)
else:
    print(f"worker {WORKER}: RESUMED at batch {state['batch']}, "
          f"{state['elapsed'] / 3600:.2f} h used, {state['n_mech']} mechanisms", flush=True)


def write_summary():
    hv = [hypervolume(state["fronts"][t]["F"], REFERENCE_POINTS[t]) for t in range(N_TARGETS)]
    summary = {
        "worker": WORKER, "batch": state["batch"], "hours": round(state["elapsed"] / 3600, 3),
        "mechanisms": state["n_mech"], "curves": state["n_curves"], "official_evals": state["n_official"],
        "hypervolume": hv,
        "front_size": [len(f["F"]) for f in state["fronts"]],
        "best_distance": [float(s["F"][0, 0]) if len(s["F"]) else None for s in state["seeds"]],
    }
    tmp = SUMMARY + ".tmp"
    with open(tmp, "w") as f:
        json.dump(summary, f, indent=1)
    os.replace(tmp, SUMMARY)
    return hv


def checkpoint():
    hv = write_summary()
    state["history"].append((state["elapsed"], state["n_mech"], *hv))
    ckpt.save(CKPT, state)
    print(f"[{state['elapsed'] / 3600:6.2f} h] batch {state['batch']:6d}  mech {state['n_mech']:9d}  "
          f"HV {hv[0]:.3f} {hv[1]:.3f} {hv[2]:.3f}  "
          f"front {[len(f['F']) for f in state['fronts']]}", flush=True)


# ---------------------------------------------------------------- main loop
budget = args.hours * 3600
save_due = ckpt.Every(60)
while state["elapsed"] < budget and not stop["flag"]:
    t0 = time.monotonic()
    b = state["batch"]
    np.random.seed((args.seed * 1_000_003 + WORKER * 7919 + b * 104_729) % (2 ** 32))

    skels = [make_skeleton() for _ in range(N_SKEL)]
    x0s = np.zeros((B, MAX_N, 2))
    As = np.zeros((B, MAX_N, MAX_N))
    types = np.ones((B, MAX_N, 1))
    mat1 = np.zeros((B, MAX_N))                       # pruned material at scale 1
    cand = np.zeros((B, MAX_N), dtype=bool)
    for i, sk in enumerate(skels):
        rows = slice(i * N_POS, (i + 1) * N_POS)
        x = np.random.uniform(0, 1, (N_POS, sk["n"], 2))
        x0s[rows, : sk["n"]] = x
        As[rows, sk["par"], sk["child"]] = 1.0
        As[rows, sk["child"], sk["par"]] = 1.0
        types[rows, : sk["n"], 0] = sk["is_fixed"][: sk["n"]]
        lengths = np.linalg.norm(x[:, sk["par"]] - x[:, sk["child"]], axis=-1)   # [N_POS, E]
        mat1[rows] = lengths @ sk["needs"].T
        cand[rows, : sk["n"]] = ~sk["is_fixed"][: sk["n"]]

    A, s = solve_and_screen(As, x0s, types)
    A, s = np.array(A), np.array(s)
    A[~cand] = np.nan
    state["n_curves"] += int(np.isfinite(A[..., 0]).sum())

    for t in range(N_TARGETS):
        ref = REFERENCE_POINTS[t]
        At = A[..., t].ravel()
        d_hat = 2 * np.pi * np.sqrt(np.clip(At, 0, None))        # upper bound on distance at the LS scale
        scale = s[..., t].ravel() * S_TARGET[t]
        m_hat = scale * mat1.ravel()
        ok = np.where(np.isfinite(At) & (d_hat < 1.3 * ref[0]) & (0.72 * m_hat < ref[1]))[0]
        if len(ok) == 0:
            continue
        by_shape = ok[np.argsort(At[ok])[: TOP_K // 2]]
        rest = np.setdiff1d(ok, by_shape)
        gain = np.clip(ref[0] - d_hat[rest], 0.02, None) * np.clip(ref[1] - m_hat[rest], 0, None)
        by_gain = rest[np.argsort(-gain)[: TOP_K - len(by_shape)]]
        chosen = np.concatenate([by_shape, by_gain])

        mechs = []
        for c in chosen:
            mi, k = divmod(int(c), MAX_N)
            sk = skels[mi // N_POS]
            for g in SCALE_GRID:
                mechs.append(pruned(sk, x0s[mi], k, scale[c] * g))
        d, m = official(mechs, targets[t])
        state["n_official"] += len(mechs)
        F = np.stack([d, m], 1)
        good = np.isfinite(F).all(1)

        inside = np.where(good & (F[:, 0] < ref[0]) & (F[:, 1] < ref[1]))[0]
        if len(inside):
            update_front(state["fronts"][t], F[inside], [mechs[i] for i in inside])

        Fc = np.where(good[:, None], F, np.inf).reshape(len(chosen), len(SCALE_GRID), 2)
        best = Fc[..., 0].argmin(1)
        rows = np.arange(len(chosen))
        seedF = Fc[rows, best]
        fin = np.where(np.isfinite(seedF[:, 0]))[0]
        if len(fin):
            update_seeds(state["seeds"][t], seedF[fin],
                         [mechs[r * len(SCALE_GRID) + best[r]] for r in fin])

    state["batch"] = b + 1
    state["n_mech"] += B
    state["elapsed"] += time.monotonic() - t0
    if save_due() or b == 0:
        checkpoint()

checkpoint()
print("STOPPED on signal (will resume)" if stop["flag"] else "DONE: budget used", flush=True)
