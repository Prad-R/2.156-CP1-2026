"""Grow good designs one dyad at a time, then re-optimize (one target per task).

Each generation:
  1. Propose: every parent gets many children, each made by attaching one new
     traced joint with two links (a dyad). Three kinds of attachment:
       rigid  - to the traced joint and one of its own parents: a new point on
                the same rigid body, placed near the old traced joint, so the
                child starts close to its parent;
       dyad   - to any two existing joints (not both ground);
       ground - to one moving joint and a brand-new ground joint.
     Children are pruned to the joints that drive the new traced joint, solved,
     resized to the target and scored with the official metric.
  2. Refine: the best children (a few per parent) get Adam steps on
     distance + w * material; every in-limits iterate is offered to the front.
  3. Select: parents for the next generation are the most accurate designs per
     joint count from parents and refined children together.

Preempt-safe: checkpoint at every stage boundary and every minute inside the
refinement. Proposals are seeded by (target, generation), so a redo is exact.

    sbatch -J grow --array=0-2 -c 8 --mem=24G -t 12:00:00 \
        -o logs/%x-%A_%a.out slurm/run.sbatch grow.py
"""
import argparse
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
from LINKS.Optimization._utils import material, unscaled_distance
from slurm import ckpt
import publish_best

MAX_N, T = 20, 200
CHUNK = 256
SIZE_BINS = [4, 5, 6, 7, 8, 9, 10, 12, 14, 16, 20]      # parents are kept per joint-count bin

p = argparse.ArgumentParser()
p.add_argument("--target", type=int, default=int(os.environ.get("SLURM_ARRAY_TASK_ID", 0)))
p.add_argument("--gens", type=int, default=12)
p.add_argument("--parents", type=int, default=44)
p.add_argument("--children", type=int, default=96, help="proposals per parent")
p.add_argument("--keep", type=int, default=512, help="children refined per generation")
p.add_argument("--per-parent", type=int, default=16)
p.add_argument("--steps", type=int, default=250)
p.add_argument("--lr", type=float, default=0.004)
p.add_argument("--refine-dir", default="checkpoints/refine")
p.add_argument("--out", default="checkpoints/grow")
args = p.parse_args()

TGT = args.target
REF = REFERENCE_POINTS[TGT]
CKPT = f"{args.out}/target_{TGT}.pkl"
os.makedirs(args.out, exist_ok=True)
target = np.load("kangaroo_target_curves.npy")[TGT]
THETAS = np.linspace(0.0, 2 * np.pi, T)
_t = np.array(equisample(jnp.asarray(target[None]), T))[0]
S_TARGET = float(np.sqrt(((_t - _t.mean(0)) ** 2).sum(-1).mean()))

stop = {"flag": False}
signal.signal(signal.SIGTERM, lambda *_: stop.update(flag=True))


# ---------------------------------------------------------------- jitted kernels
@jax.jit
def traced_radius(x0s, As, types, tidx):
    """rms radius of each traced curve (nan if the mechanism does not assemble)."""
    sol = dyadic_solve(As, x0s, types, THETAS)
    ok = jnp.isfinite(sol).all(axis=(1, 2, 3))
    c = sol[jnp.arange(sol.shape[0]), tidx]
    c = equisample(jnp.where(ok[:, None, None], c, jnp.asarray(_t, dtype=c.dtype)), T)
    c = c - c.mean(1, keepdims=True)
    return jnp.where(ok, jnp.sqrt((c ** 2).sum(-1).mean(-1)), jnp.nan)


@jax.jit
def score(x0s, As, types, tidx):
    _, d = unscaled_distance(x0s, As, types, THETAS, tidx, jnp.broadcast_to(target, (x0s.shape[0], T, 2)))
    _, m = material(x0s, As)
    return d, m


@jax.jit
def value_and_grad(x0s, As, types, tidx, w):
    def loss(x):
        _, d = unscaled_distance(x, As, types, THETAS, tidx, jnp.broadcast_to(target, (x.shape[0], T, 2)))
        _, m = material(x, As)
        return (jnp.where(jnp.isfinite(d), d, 0.0) + w * m).sum(), (d, m)
    (_, (d, m)), g = jax.value_and_grad(loss, has_aux=True)(x0s)
    return d, m, g


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


def batched(fn, mechs, n_out):
    """Run a jitted kernel over mechs in fixed-size chunks."""
    outs = [[] for _ in range(n_out)]
    for c in range(0, len(mechs), CHUNK):
        chunk = mechs[c:c + CHUNK]
        pad = chunk + [chunk[0]] * (CHUNK - len(chunk))
        res = fn(*pack(pad))
        res = res if isinstance(res, tuple) else (res,)
        for o, r in zip(outs, res):
            o.append(np.array(r)[: len(chunk)])
    return [np.concatenate(o) for o in outs]


def official(mechs):
    d, m = batched(score, mechs, 2)
    return np.stack([d, m], 1)


# ---------------------------------------------------------------- structure edits
def make(x0, edges, fixed):
    """Mechanism dict pruned to the ancestors of its last joint, in solver order.

    Input must already be in solver order (ground 0, crank 1, other grounds,
    then moving joints each after both of its parents) with the traced joint last.
    """
    n = len(x0)
    is_fixed = np.zeros(n, dtype=bool)
    is_fixed[fixed] = True
    par, child = edges.min(1), edges.max(1)
    need = np.zeros(n, dtype=bool)
    need[[0, 1, n - 1]] = True
    for k in range(n - 1, 1, -1):                       # parents have lower indices
        if need[k] and not is_fixed[k]:
            need[par[child == k]] = True
    keep = np.where(need)[0]
    remap = -np.ones(n, dtype=int)
    remap[keep] = np.arange(len(keep))
    e = need[child] & need[par] & ~is_fixed[child]
    return {"x0": np.asarray(x0, dtype=np.float64)[keep],
            "edges": np.stack([remap[par[e]], remap[child[e]]], 1),
            "fixed_joints": remap[keep[is_fixed[keep]]],
            "motor": np.array([0, 1]), "target_joint": len(keep) - 1}


def propose(parent, rng):
    """One child: the parent plus one new traced joint hung from two joints."""
    x0, edges, fixed = parent["x0"], parent["edges"], parent["fixed_joints"]
    n = len(x0)
    if n >= MAX_N:
        return None
    is_fixed = np.zeros(n, dtype=bool)
    is_fixed[fixed] = True
    moving = np.where(~is_fixed)[0]
    a = n - 1                                            # current traced joint
    size = np.sqrt(((x0 - x0.mean(0)) ** 2).sum(-1).mean())
    lo, hi = x0.min(0) - 0.6 * size, x0.max(0) + 0.6 * size
    kind = rng.choice(["rigid", "dyad", "ground"], p=[0.3, 0.45, 0.25])

    if kind == "rigid" and n > 2:
        par = edges.min(1)[edges.max(1) == a]
        b = rng.choice(par)
        c = x0[a] + rng.normal(scale=0.35 * size, size=2)
        return make(np.vstack([x0, c]), np.vstack([edges, [a, n], [b, n]]), fixed)

    if kind == "ground" and n + 2 <= MAX_N:
        u = a if rng.random() < 0.6 else rng.choice(moving)
        g, c = rng.uniform(lo, hi), rng.uniform(lo, hi)
        f_end = int(is_fixed.sum()) + 1                  # first moving index after the crank (index 1)
        shift = lambda i: i + (np.asarray(i) >= f_end)   # new ground slots in at f_end
        x_new = np.vstack([x0[:f_end], g, x0[f_end:], c])
        e_new = np.vstack([shift(edges), [f_end, n + 1], [shift(u), n + 1]])
        return make(x_new, e_new, np.append(shift(fixed), f_end))

    u = a if rng.random() < 0.6 else rng.choice(moving)
    v = rng.choice(np.setdiff1d(np.arange(n), [u]))
    c = rng.uniform(lo, hi)
    return make(np.vstack([x0, c]), np.vstack([edges, [u, n], [v, n]]), fixed)


# ---------------------------------------------------------------- front and selection
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


def offer(front, F_new, mech_fn):
    """Add in-limits points to the front; mech_fn(i) builds the mechanism for new row i."""
    inside = np.where(np.isfinite(F_new).all(1) & (F_new[:, 0] < REF[0]) & (F_new[:, 1] < REF[1]))[0]
    if not len(inside):
        return
    F = np.concatenate([front["F"], F_new[inside]])
    idx = nondominated(F)
    if len(idx) > 3000:
        idx = idx[np.linspace(0, len(idx) - 1, 3000).round().astype(int)]
    n_old = len(front["F"])
    front["mechs"] = [front["mechs"][i] if i < n_old else mech_fn(inside[i - n_old]) for i in idx]
    front["F"] = F[idx]


def select_parents(mechs, F):
    """Most accurate designs per joint-count bin, round-robin until the quota is filled."""
    sizes = np.array([len(m["x0"]) for m in mechs])
    bins = np.searchsorted(SIZE_BINS, sizes)
    ranked = [list(np.where(bins == b)[0][np.argsort(F[bins == b, 0])]) for b in range(len(SIZE_BINS) + 1)]
    chosen = []
    while len(chosen) < args.parents and any(ranked):
        for r in ranked:
            if r and len(chosen) < args.parents:
                chosen.append(r.pop(0))
    return [mechs[i] for i in chosen], F[chosen]


def with_x(m, x):
    return {**m, "x0": np.array(x[: len(m["x0"])], dtype=np.float64)}


# ---------------------------------------------------------------- state
state = ckpt.load(CKPT)
if state is None:
    src = ckpt.load(f"{args.refine_dir}/target_{TGT}.pkl")
    if src is None:
        raise SystemExit(f"no refinement checkpoint for target {TGT} in {args.refine_dir}")
    pool = [with_x(src["mechs"][s], x) for x, s in zip(src["front"]["x"], src["front"]["src"])]
    pool += [with_x(m, x) for m, x in zip(src["mechs"], src["last_good"])]
    pool = [m for m in pool if len(m["x0"]) >= 4]
    F = official(pool)
    ok = np.isfinite(F).all(1)
    pool, F = [m for m, k in zip(pool, ok) if k], F[ok]
    parents, pF = select_parents(pool, F)
    state = {"target": TGT, "gen": 0, "stage": "propose", "elapsed": 0.0,
             "parents": parents, "parents_F": pF, "front": {"F": np.zeros((0, 2)), "mechs": []},
             "refine": None, "history": []}
    offer(state["front"], F, lambda i: pool[i])
    print(f"target {TGT}: fresh start from {len(pool)} refined designs; {len(parents)} parents, "
          f"best distance {pF[:, 0].min():.4f}, front HV {hypervolume(state['front']['F']):.3f}", flush=True)
else:
    print(f"target {TGT}: RESUMED at generation {state['gen']} ({state['stage']}), "
          f"{state['elapsed'] / 3600:.2f} h used", flush=True)

B1, B2, EPS = 0.9, 0.999, 1e-8
save_due = ckpt.Every(60)


def log(msg):
    F = state["front"]["F"]
    best = F[:, 0].min() if len(F) else float("nan")
    print(f"[{state['elapsed'] / 3600:5.2f} h] gen {state['gen']:2d} {msg}  |  front HV {hypervolume(F):.3f}, "
          f"{len(F)} designs, best distance {best:.4f}", flush=True)


while state["gen"] < args.gens and not stop["flag"]:
    t0 = time.monotonic()

    if state["stage"] == "propose":
        rng = np.random.default_rng([2156, TGT, state["gen"]])
        kids, owner = [], []
        for pi, par in enumerate(state["parents"]):
            for _ in range(args.children):
                k = propose(par, rng)
                if k is not None and len(k["x0"]) >= 4:
                    kids.append(k)
                    owner.append(pi)
        owner = np.array(owner)
        (radius,) = batched(traced_radius, kids, 1)
        alive = np.where(np.isfinite(radius) & (radius > 1e-6))[0]
        kids = [with_x(kids[i], kids[i]["x0"] * (S_TARGET / radius[i])) for i in alive]   # resize to the target
        owner = owner[alive]
        F = official(kids)
        ok = np.isfinite(F).all(1)
        order = [i for i in np.argsort(np.where(ok, F[:, 0], np.inf)) if ok[i]]
        taken, count = [], np.zeros(len(state["parents"]), dtype=int)
        for i in order:                                   # best first, a few per parent
            if count[owner[i]] < args.per_parent and len(taken) < args.keep:
                taken.append(i)
                count[owner[i]] += 1
        if not taken:                                     # nothing assembled: try again with new proposals
            state["gen"] += 1
            state["elapsed"] += time.monotonic() - t0
            ckpt.save(CKPT, state)
            log("no working children")
            continue
        members = [kids[i] for i in taken]
        offer(state["front"], F[taken], lambda i: members[i])
        pad = (-len(members)) % CHUNK
        members += members[:pad]
        x0, A, types, tidx = pack(members)
        size = np.array([np.sqrt(((m["x0"] - m["x0"].mean(0)) ** 2).sum(-1).mean()) for m in members])
        state["refine"] = {
            "mechs": members, "step": 0, "x": x0, "A": A, "types": types, "tidx": tidx,
            "real": np.arange(MAX_N)[None, :] < np.array([len(m["x0"]) for m in members])[:, None],
            "w": np.where(np.arange(len(members)) % 2 == 0, 0.0, 0.05),
            "adam_m": np.zeros_like(x0), "adam_v": np.zeros_like(x0), "lr": args.lr * size,
            "last_good": x0.copy(), "last_F": np.full((len(members), 2), np.inf),
            "n_proposed": len(alive), "start_best": float(F[taken, 0].min()) if taken else float("nan")}
        state["stage"] = "refine"
        state["elapsed"] += time.monotonic() - t0
        ckpt.save(CKPT, state)
        log(f"proposed {len(alive)} working children, refining {len(taken)} (best child distance "
            f"{state['refine']['start_best']:.4f}, best parent {state['parents_F'][:, 0].min():.4f})")
        continue

    # ---- refine stage
    r = state["refine"]
    N = len(r["x"])
    while r["step"] < args.steps and not stop["flag"]:
        t0 = time.monotonic()
        step = r["step"] + 1
        decay = 0.3 ** (r["step"] / args.steps)
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
        offer(state["front"], F_step, lambda i: with_x(r["mechs"][i], x_before[i]))
        r["step"] = step
        state["elapsed"] += time.monotonic() - t0
        if save_due():
            ckpt.save(CKPT, state)
            log(f"refine step {step:3d}/{args.steps}")
    if r["step"] < args.steps:
        break                                              # stopped by signal mid-refinement

    # ---- select the next parents from parents and refined children
    t0 = time.monotonic()
    kids = [with_x(m, x) for m, x in zip(r["mechs"], r["last_good"])]
    ok = np.isfinite(r["last_F"]).all(1)
    pool = state["parents"] + [k for k, o in zip(kids, ok) if o]
    F = np.concatenate([state["parents_F"], r["last_F"][ok]])
    improved = int((r["last_F"][ok, 0] < state["parents_F"][:, 0].min()).sum())
    state["parents"], state["parents_F"] = select_parents(pool, F)
    sizes = [len(m["x0"]) for m in state["parents"]]
    state["history"].append((state["gen"], state["elapsed"], hypervolume(state["front"]["F"]),
                             float(state["parents_F"][:, 0].min()), int(np.max(sizes))))
    state["gen"] += 1
    state["stage"], state["refine"] = "propose", None
    state["elapsed"] += time.monotonic() - t0
    ckpt.save(CKPT, state)
    best = int(np.argmin(state["parents_F"][:, 0]))
    log(f"done: {improved} children beat the best parent; best design now distance "
        f"{state['parents_F'][best, 0]:.4f} with {sizes[best]} joints; parent sizes {min(sizes)}-{max(sizes)}")
    publish_best.merge_and_publish({f"Problem {TGT + 1}": state["front"]["mechs"]},
                                   f"grow target {TGT + 1}, generation {state['gen']}")

ckpt.save(CKPT, state)
print("STOPPED on signal (will resume)" if stop["flag"] else "DONE", flush=True)
