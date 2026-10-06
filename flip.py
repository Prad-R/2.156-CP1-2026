"""Try the other assembly branch of each joint, then refine (one target per task).

Every moving joint sits on one side of the line through its two parents. Put
it on the other side, keep every link length, and re-assemble the joints that
hang off it: the same links trace a different curve. Gradient descent cannot
get there, because the mechanism would have to pass through a locked position.

For each parent (taken from all along the current best front) this enumerates
the flips (all combinations for small designs; singles, pairs and a random
sample for larger ones), scores them with the official metric, and refines the
best few per parent. Material is unchanged by a flip.

    sbatch -p mit_preemptable,mit_normal -J flip --array=0-2 -c 8 --mem=16G -t 03:00:00 \
        -o logs/%x-%A_%a.out slurm/run.sbatch flip.py
"""
import argparse
import itertools
import os
import signal
import time

import numpy as np

from LINKS.CP import REFERENCE_POINTS
from slurm import ckpt
import linkcore as lc
import publish_best

p = argparse.ArgumentParser()
p.add_argument("--target", type=int, default=int(os.environ.get("SLURM_ARRAY_TASK_ID", 0)))
p.add_argument("--n-front", type=int, default=64)
p.add_argument("--n-accurate", type=int, default=64)
p.add_argument("--max-children", type=int, default=160, help="flip combinations tried per parent")
p.add_argument("--per-parent", type=int, default=8, help="children refined per parent")
p.add_argument("--keep", type=int, default=768)
p.add_argument("--steps", type=int, default=400)
p.add_argument("--lr", type=float, default=0.004)
p.add_argument("--out", default="checkpoints/flip")
args = p.parse_args()

TGT = args.target
REF = REFERENCE_POINTS[TGT]
CKPT = f"{args.out}/target_{TGT}.pkl"
os.makedirs(args.out, exist_ok=True)
target = np.load("kangaroo_target_curves.npy")[TGT].astype(np.float64)
S_TARGET = lc.target_radius(target)

stop = {"flag": False}
signal.signal(signal.SIGTERM, lambda *_: stop.update(flag=True))


def dyad_joints(m):
    """Moving joints other than the crank, with their two parents."""
    n = len(m["x0"])
    is_fixed = np.zeros(n, dtype=bool)
    is_fixed[m["fixed_joints"]] = True
    par, child = m["edges"].min(1), m["edges"].max(1)
    return [(k, *par[child == k]) for k in range(2, n) if not is_fixed[k] and (child == k).sum() == 2]


def reassemble(m, joints, flipped):
    """Same link lengths, with the joints in `flipped` on the other side of their parents' line."""
    old, new = m["x0"], m["x0"].copy()
    for k, i, j in joints:
        li, lj = np.linalg.norm(old[k] - old[i]), np.linalg.norm(old[k] - old[j])
        base = old[j] - old[i]
        side = np.sign(base[0] * (old[k] - old[i])[1] - base[1] * (old[k] - old[i])[0]) or 1.0
        if k in flipped:
            side = -side
        dvec = new[j] - new[i]
        D = np.linalg.norm(dvec)
        if D < 1e-12 or D > li + lj or D < abs(li - lj):
            return None                                   # the links no longer reach each other
        a = (li ** 2 - lj ** 2 + D ** 2) / (2 * D)
        h = np.sqrt(max(li ** 2 - a ** 2, 0.0))
        new[k] = new[i] + a * dvec / D + side * h * np.array([-dvec[1], dvec[0]]) / D
    return lc.with_x(m, new)


def flip_sets(joints, rng):
    ks = [k for k, _, _ in joints]
    if len(ks) <= 7:
        return [set(c) for r in range(1, len(ks) + 1) for c in itertools.combinations(ks, r)]
    sets = [set(c) for r in (1, 2) for c in itertools.combinations(ks, r)]
    while len(sets) < args.max_children:
        sets.append(set(rng.choice(ks, size=rng.integers(3, len(ks) + 1), replace=False).tolist()))
    return sets[: args.max_children]


# ---------------------------------------------------------------- state
state = ckpt.load(CKPT)
if state is None:
    t0 = time.monotonic()
    pool = np.load(publish_best.NPY, allow_pickle=True).item()[f"Problem {TGT + 1}"]
    pool = [m for m in pool if len(m["x0"]) >= 4]
    d, mat = lc.batched(lc.score, pool, 2, target)
    F = np.stack([d, mat], 1)
    chosen = list(dict.fromkeys(lc.spread_along_front(F, args.n_front) + lc.pick_per_size(pool, d, args.n_accurate)))
    parents, pF = [pool[i] for i in chosen], F[chosen]

    rng = np.random.default_rng([2156, TGT, 7])
    kids, owner = [], []
    for pi, par in enumerate(parents):
        joints = dyad_joints(par)
        for fs in flip_sets(joints, rng):
            k = reassemble(par, joints, fs)
            if k is not None:
                kids.append(k)
                owner.append(pi)
    owner = np.array(owner)
    (radius,) = lc.batched(lc.traced_radius, kids, 1)
    alive = np.where(np.isfinite(radius) & (radius > 1e-6))[0]
    kids = [lc.with_x(kids[i], kids[i]["x0"] * (S_TARGET / radius[i])) for i in alive]    # resize to the target
    owner = owner[alive]
    d, mat = lc.batched(lc.score, kids, 2, target)
    kF = np.stack([d, mat], 1)
    ok = np.isfinite(kF).all(1) & (kF[:, 1] < REF[1])
    beats = ok & (kF[:, 0] < pF[owner, 0])
    order = [i for i in np.argsort(np.where(ok, kF[:, 0] / pF[owner, 0], np.inf)) if ok[i]]   # best relative to parent first
    taken, count = [], np.zeros(len(parents), dtype=int)
    for i in order:
        if count[owner[i]] < args.per_parent and len(taken) < args.keep:
            taken.append(i)
            count[owner[i]] += 1
    members = [kids[i] for i in taken]
    state = {"target": TGT, "elapsed": time.monotonic() - t0, "front": lc.empty_front(),
             "parents_F": pF, "owner": owner[taken],
             "run": lc.new_run(members, [0.0 if i % 2 == 0 else 0.08 for i in range(len(members))], args.lr),
             "start_F": kF[taken], "history": [],
             "stats": {"parents": len(parents), "flips_tried": int(len(owner) + 0), "assembled": int(len(alive)),
                       "valid": int(ok.sum()), "beat_parent_before_refining": int(beats.sum()),
                       "parents_with_a_better_flip": int(len(set(owner[beats])))}}
    lc.offer(state["front"], F, lambda i: pool[i], REF)
    lc.offer(state["front"], kF[taken], lambda i: members[i], REF)
    ckpt.save(CKPT, state)
    print(f"target {TGT}: {state['stats']}; refining {len(members)} flipped designs; "
          f"front HV {lc.hypervolume(state['front']['F'], REF):.3f}", flush=True)
else:
    print(f"target {TGT}: RESUMED at step {state['run']['step']}, {state['elapsed'] / 3600:.2f} h used", flush=True)

r = state["run"]
N = len(r["x"])
save_due = ckpt.Every(60)


def log():
    F = state["front"]["F"]
    n = r["n_real"]
    won = r["last_F"][:n, 0] < state["parents_F"][state["owner"], 0]
    state["history"].append((r["step"], state["elapsed"], lc.hypervolume(F, REF), int(won.sum())))
    print(f"[{state['elapsed'] / 3600:5.2f} h] step {r['step']:4d}/{args.steps}  front HV {lc.hypervolume(F, REF):.3f} "
          f"({len(F)} designs), best distance {F[:, 0].min():.4f}  |  flipped designs now more accurate than "
          f"their parent: {int(won.sum())}/{n}, from {len(set(state['owner'][won]))} parents", flush=True)


while r["step"] < args.steps and not stop["flag"]:
    t0 = time.monotonic()
    step = r["step"] + 1
    x_before = r["x"].copy()
    F_step = np.full((N, 2), np.inf)
    for c in range(0, N, lc.CHUNK):
        sl = slice(c, c + lc.CHUNK)
        d, m, g = lc.value_and_grad(r["x"][sl], r["A"][sl], r["types"][sl], r["tidx"][sl], r["w"][sl], target)
        F_step[sl] = lc.adam_update(r, sl, d, m, g, step, 0.3 ** (r["step"] / args.steps))
    lc.offer(state["front"], F_step, lambda i: lc.with_x(r["mechs"][i], x_before[i]), REF)
    r["step"] = step
    state["elapsed"] += time.monotonic() - t0
    if save_due() or step == 1:
        log()
        ckpt.save(CKPT, state)

log()
ckpt.save(CKPT, state)
publish_best.merge_and_publish({f"Problem {TGT + 1}": state["front"]["mechs"]}, f"flip target {TGT + 1}, step {r['step']}")
print("STOPPED on signal (will resume)" if stop["flag"] else "DONE", flush=True)
