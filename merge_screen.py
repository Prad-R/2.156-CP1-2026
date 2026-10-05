"""Merge the screening workers' checkpoints into one scored submission.

Read-only on the checkpoints, so it is safe to run while workers are active.
Every merge is kept: results/<name>/<timestamp>/ holds submission.npy,
summary.json, fronts.png and best_<k>.png (drawings of the best mechanisms for
each target); results/<name>/latest/ is a copy of the newest merge.

    sbatch -J merge -c 4 --mem=8G -t 00:20:00 slurm/run.sbatch merge_screen.py
"""
import argparse
import glob
import json
import os
import shutil
import time

os.environ["JAX_PLATFORMS"] = "cpu"

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from LINKS.CP import REFERENCE_POINTS, SCORE_NORMALIZERS, evaluate_submission
from LINKS.Geometry import CurveEngine
from LINKS.Kinematics import MechanismSolver
from LINKS.Visualization import MechanismVisualizer
from slurm import ckpt

p = argparse.ArgumentParser()
p.add_argument("--ckpt-dir", default="checkpoints/screen")
p.add_argument("--name", default="screen")
p.add_argument("--out", default="results")
args = p.parse_args()

NAMES = ["Kangaroo 1", "Kangaroo 2", "Kangaroo 3"]
MAX_PER_PROBLEM = 1000
STAMP = time.strftime("%Y%m%d-%H%M%S")
OUT = f"{args.out}/{args.name}/{STAMP}"
os.makedirs(OUT, exist_ok=True)
targets = np.load("kangaroo_target_curves.npy")


def nondominated(F):
    order = np.lexsort((F[:, 1], F[:, 0]))
    best_m = np.minimum.accumulate(F[order, 1])
    keep = np.ones(len(F), dtype=bool)
    keep[1:] = F[order, 1][1:] < best_m[:-1]
    return order[keep]


states = [s for s in (ckpt.load(f) for f in sorted(glob.glob(f"{args.ckpt_dir}/worker_*.pkl"))) if s]
if not states:
    raise SystemExit(f"no checkpoints in {args.ckpt_dir}")

submission, fronts = {}, []
for t in range(len(REFERENCE_POINTS)):
    F = np.concatenate([s["fronts"][t]["F"] for s in states])
    mechs = sum((s["fronts"][t]["mechs"] for s in states), [])
    if len(F):
        idx = nondominated(F)
        if len(idx) > MAX_PER_PROBLEM:
            idx = idx[np.linspace(0, len(idx) - 1, MAX_PER_PROBLEM).round().astype(int)]
        F, mechs = F[idx], [mechs[i] for i in idx]
    fronts.append(F)
    submission[f"Problem {t + 1}"] = mechs

np.save(f"{OUT}/submission.npy", submission)
score = evaluate_submission(submission)

summary = {
    "merged_at": STAMP,
    "workers": len(states),
    "worker_hours": round(sum(s["elapsed"] for s in states) / 3600, 2),
    "mechanisms_sampled": int(sum(s["n_mech"] for s in states)),
    "curves_screened": int(sum(s["n_curves"] for s in states)),
    "official_evaluations": int(sum(s["n_official"] for s in states)),
    "score": score,
    "front_size": [len(F) for F in fronts],
    "best_distance": [float(F[:, 0].min()) if len(F) else None for F in fronts],
    "least_material": [float(F[:, 1].min()) if len(F) else None for F in fronts],
    "seed_best_distance": [float(min(s["seeds"][t]["F"][0, 0] for s in states if len(s["seeds"][t]["F"])))
                           if any(len(s["seeds"][t]["F"]) for s in states) else None
                           for t in range(len(REFERENCE_POINTS))],
}
with open(f"{OUT}/summary.json", "w") as f:
    json.dump(summary, f, indent=1)
print(json.dumps(summary, indent=1))

# ---- figures: fronts in their boxes, and the most accurate design per target
fig, axs = plt.subplots(1, 3, figsize=(15, 4.6))
for t, ax in enumerate(axs):
    ref, F = REFERENCE_POINTS[t], fronts[t]
    ax.set_xlim(0, ref[0]); ax.set_ylim(0, ref[1])
    if len(F):
        o = np.argsort(F[:, 0])
        xs = np.append(F[o, 0], ref[0])
        ax.fill_between(xs, np.append(F[o, 1], F[o[-1], 1]), ref[1], step="post", color="#1f5fa8", alpha=0.22)
        ax.plot(F[o, 0], F[o, 1], ".", color="#16202e", ms=4)
    hv = score["Score Breakdown"][f"Problem {t + 1}"]
    ax.set_title(f"{NAMES[t]}: hypervolume {hv:.2f} (normalized {hv / SCORE_NORMALIZERS[t]:.2f})")
    ax.set_xlabel("Distance"); ax.set_ylabel("Material")
fig.tight_layout()
fig.savefig(f"{OUT}/fronts.png", dpi=130)
plt.close(fig)

# ---- drawings: for each target, the most accurate, best-balanced and lightest designs
solver = MechanismSolver(device="cpu")
engine = CurveEngine(device="cpu")
visualizer = MechanismVisualizer()
for t in range(len(REFERENCE_POINTS)):
    F, ref = fronts[t], REFERENCE_POINTS[t]
    if not len(F):
        continue
    picks = [("Most accurate", int(np.argmin(F[:, 0]))),
             ("Best balanced", int(np.argmax((ref[0] - F[:, 0]) * (ref[1] - F[:, 1])))),
             ("Least material", int(np.argmin(F[:, 1])))]
    merged = {}                                   # one row per distinct design
    for label, i in picks:
        merged[i] = f"{merged[i]} and {label.lower()}" if i in merged else label
    picks = [(label, i) for i, label in merged.items()]
    fig, axs = plt.subplots(len(picks), 2, figsize=(11, 5 * len(picks)), squeeze=False)
    for row, (label, i) in enumerate(picks):
        m = submission[f"Problem {t + 1}"][i]
        visualizer(m["x0"], m["edges"], m["fixed_joints"], m["motor"], highlight=m["target_joint"], ax=axs[row, 0])
        axs[row, 0].set_title(f"{label}: distance {F[i, 0]:.3f}, material {F[i, 1]:.2f}, {len(m['x0'])} joints")
        traced = solver(m["x0"], m["edges"], m["fixed_joints"], m["motor"])[m["target_joint"]]
        engine.visualize_single_comparison(traced, targets[t], ax=axs[row, 1])
        axs[row, 1].set_title("Traced curve (orange) on target (blue)")
    fig.suptitle(f"{NAMES[t]} — merge {STAMP}", fontsize=15)
    fig.tight_layout()
    fig.savefig(f"{OUT}/best_{t + 1}.png", dpi=120)
    plt.close("all")

latest = f"{args.out}/{args.name}/latest"
shutil.rmtree(latest, ignore_errors=True)
shutil.copytree(OUT, latest)
print(f"saved {OUT} (also copied to {latest})")
