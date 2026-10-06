"""Figures and a summary for the current best submission and the optimization runs.

Writes results/best/<timestamp>/ (mirrored to results/best/latest/):
  summary.json      official score of submissions/best_submission.npy, sizes, best distances
  fronts.png        the three Pareto fronts inside their limit boxes
  best_<k>.png      drawings of the most accurate, best-balanced and lightest designs
  progress.png      hypervolume and best distance over refinement steps / growth generations

    sbatch -J report -c 4 --mem=8G -t 00:20:00 slurm/run.sbatch report_best.py
"""
import json
import os
import shutil
import time
from collections import Counter

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
import publish_best

NAMES = ["Kangaroo 1", "Kangaroo 2", "Kangaroo 3"]
STAMP = time.strftime("%Y%m%d-%H%M%S")
OUT = f"results/best/{STAMP}"
os.makedirs(OUT, exist_ok=True)
targets = np.load("kangaroo_target_curves.npy")

submission = np.load(publish_best.NPY, allow_pickle=True).item()
score = evaluate_submission(submission)
fronts = [publish_best._official(submission[k], targets[t]) if submission[k] else np.zeros((0, 2))
          for t, k in enumerate(publish_best.PROBLEMS)]

with open(publish_best.META) as f:
    meta = json.load(f)
summary = {
    "reported_at": STAMP, "score": score, "source": meta["source"],
    "designs": [len(F) for F in fronts],
    "best_distance": [float(F[:, 0].min()) if len(F) else None for F in fronts],
    "least_material": [float(F[:, 1].min()) if len(F) else None for F in fronts],
    # unclaimed area, as overall-score points: left of the most accurate design (needs better accuracy)
    # and under the front (needs the same accuracy with less material)
    "unclaimed_left_of_front": [float(F[:, 0].min() * REFERENCE_POINTS[t][1] / SCORE_NORMALIZERS[t] / 3) if len(F) else None
                                for t, F in enumerate(fronts)],
    "unclaimed_under_front": [float((REFERENCE_POINTS[t][0] * REFERENCE_POINTS[t][1] - score["Score Breakdown"][k]
                                     - F[:, 0].min() * REFERENCE_POINTS[t][1]) / SCORE_NORMALIZERS[t] / 3) if len(F) else None
                              for t, (k, F) in enumerate(zip(publish_best.PROBLEMS, fronts))],
    "joint_counts": [dict(sorted(Counter(len(m["x0"]) for m in submission[k]).items())) for k in publish_best.PROBLEMS],
}
with open(f"{OUT}/summary.json", "w") as f:
    json.dump(summary, f, indent=1)
print(json.dumps(summary, indent=1))

# ---- fronts
fig, axs = plt.subplots(1, 3, figsize=(15, 4.6))
for t, ax in enumerate(axs):
    ref, F = REFERENCE_POINTS[t], fronts[t]
    ax.set_xlim(0, ref[0]); ax.set_ylim(0, ref[1])
    if len(F):
        o = np.argsort(F[:, 0])
        ax.fill_between(np.append(F[o, 0], ref[0]), np.append(F[o, 1], F[o[-1], 1]), ref[1],
                        step="post", color="#1f5fa8", alpha=0.22)
        ax.plot(F[o, 0], F[o, 1], ".", color="#16202e", ms=3)
    hv = score["Score Breakdown"][f"Problem {t + 1}"]
    ax.set_title(f"{NAMES[t]}: hypervolume {hv:.2f} (normalized {hv / SCORE_NORMALIZERS[t]:.2f})")
    ax.set_xlabel("Distance"); ax.set_ylabel("Material")
fig.tight_layout()
fig.savefig(f"{OUT}/fronts.png", dpi=130)
plt.close(fig)

# ---- drawings of the best designs
solver = MechanismSolver(device="cpu")
engine = CurveEngine(device="cpu")
visualizer = MechanismVisualizer()
for t, k in enumerate(publish_best.PROBLEMS):
    F, ref = fronts[t], REFERENCE_POINTS[t]
    if not len(F):
        continue
    picks = [("Most accurate", int(np.argmin(F[:, 0]))),
             ("Best balanced", int(np.argmax((ref[0] - F[:, 0]) * (ref[1] - F[:, 1])))),
             ("Least material", int(np.argmin(F[:, 1])))]
    merged = {}
    for label, i in picks:
        merged[i] = f"{merged[i]} and {label.lower()}" if i in merged else label
    picks = [(label, i) for i, label in merged.items()]
    fig, axs = plt.subplots(2, len(picks), figsize=(5.5 * len(picks), 10.5), squeeze=False)
    for col, (label, i) in enumerate(picks):
        m = submission[k][i]
        visualizer(m["x0"], m["edges"], m["fixed_joints"], m["motor"], highlight=m["target_joint"], ax=axs[0, col])
        axs[0, col].set_title(f"{label}\ndistance {F[i, 0]:.3f}, material {F[i, 1]:.2f}, {len(m['x0'])} joints")
        traced = solver(m["x0"], m["edges"], m["fixed_joints"], m["motor"])[m["target_joint"]]
        engine.visualize_single_comparison(traced, targets[t], ax=axs[1, col])
        axs[1, col].set_title("Traced curve (orange) on target (blue)")
    fig.suptitle(f"{NAMES[t]} — best submission at {STAMP}", fontsize=15)
    fig.tight_layout()
    fig.savefig(f"{OUT}/best_{t + 1}.png", dpi=120)
    plt.close("all")

# ---- progress of the optimizing stages
fig, axs = plt.subplots(2, 3, figsize=(15, 7), sharex="col")
any_data = False
for t in range(3):
    r = ckpt.load(f"checkpoints/refine/target_{t}.pkl")
    g = ckpt.load(f"checkpoints/grow/target_{t}.pkl")
    x_end = 0
    if r and r["history"]:
        h = np.array(r["history"])                       # step, elapsed, hv, best distance
        axs[0, t].plot(h[:, 0], h[:, 2], color="#1f5fa8", label="refinement")
        axs[1, t].plot(h[:, 0], h[:, 3], color="#1f5fa8")
        x_end, any_data = h[-1, 0], True
    if g and g["history"]:
        h = np.array(g["history"])                       # generation, elapsed, hv, best distance, max joints
        xs = x_end + (h[:, 0] + 1) * 250                 # each generation refines for 250 steps
        axs[0, t].plot(xs, h[:, 2], "o-", color="#c8601a", ms=3, label="growth")
        axs[1, t].plot(xs, h[:, 3], "o-", color="#c8601a", ms=3)
        any_data = True
    axs[0, t].set_title(NAMES[t]); axs[1, t].set_xlabel("Optimizer steps")
    axs[0, t].legend(loc="lower right", fontsize=9)
axs[0, 0].set_ylabel("Front hypervolume (this stage's own front)")
axs[1, 0].set_ylabel("Best distance")
fig.tight_layout()
if any_data:
    fig.savefig(f"{OUT}/progress.png", dpi=130)
plt.close(fig)

latest = "results/best/latest"
shutil.rmtree(latest, ignore_errors=True)
shutil.copytree(OUT, latest)
print(f"saved {OUT} (also copied to {latest})")
