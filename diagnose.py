"""Where along each target is the distance being lost?

For designs in submissions/best_submission.npy, aligns the traced curve to the
target exactly as the scorer does and looks at the error point by point.

Writes results/diagnose/<timestamp>/ (mirrored to latest/):
  error_<k>.png   three designs per target: the target coloured by error with
                  the traced curve over it, and the error along the perimeter
  family_<k>.png  the average error profile of the 50 most accurate designs:
                  where this whole family of linkages fails
  summary.json    how concentrated the error is, and where its peaks are

    sbatch -p mit_preemptable,mit_normal -J diagnose -c 4 --mem=8G -t 00:20:00 slurm/run.sbatch diagnose.py
"""
import json
import os
import shutil
import time

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from LINKS.CP import REFERENCE_POINTS
import linkcore as lc
import publish_best

NAMES = ["Kangaroo 1", "Kangaroo 2", "Kangaroo 3"]
STAMP = time.strftime("%Y%m%d-%H%M%S")
OUT = f"results/diagnose/{STAMP}"
os.makedirs(OUT, exist_ok=True)
targets = np.load("kangaroo_target_curves.npy")
submission = np.load(publish_best.NPY, allow_pickle=True).item()
summary = {"at": STAMP, "targets": []}


def perimeter_marks(ax, tgt):
    """Label every tenth of the perimeter so the two panels can be matched up."""
    for frac in range(0, 100, 10):
        i = int(frac / 100 * lc.T)
        ax.annotate(f"{frac}", tgt[i], fontsize=8, color="#4a5568", ha="center", va="center",
                    bbox=dict(boxstyle="circle,pad=0.15", fc="white", ec="#b0b0b0", lw=0.6))


for t, key in enumerate(publish_best.PROBLEMS):
    mechs = submission[key]
    aligned, tgt, err, d = lc.batched(lc.align, mechs, 4, targets[t])
    _, m = lc.batched(lc.score, mechs, 2, targets[t])
    ref = REFERENCE_POINTS[t]
    small = np.where(np.array([len(x["x0"]) for x in mechs]) <= 6)[0]
    picks = [("Most accurate", int(np.argmin(d))),
             ("Best balanced", int(np.argmax((ref[0] - d) * (ref[1] - m)))),
             ("Most accurate with 6 joints or fewer", int(small[np.argmin(d[small])]))]
    pos = np.arange(lc.T) / lc.T * 100
    vmax = float(np.percentile(err[[i for _, i in picks]], 99))

    fig, axs = plt.subplots(len(picks), 2, figsize=(13, 4.6 * len(picks)), gridspec_kw={"width_ratios": [1, 1.5]})
    rows = []
    for row, (label, i) in enumerate(picks):
        ax = axs[row, 0]
        ax.plot(aligned[i, :, 0], aligned[i, :, 1], color="#16202e", lw=1.2, label="traced curve")
        sc = ax.scatter(tgt[i, :, 0], tgt[i, :, 1], c=err[i], cmap="YlOrRd", vmin=0, vmax=vmax, s=22, zorder=3,
                        edgecolors="#8a8a8a", linewidths=0.3)
        perimeter_marks(ax, tgt[i] * 1.12)
        ax.set_aspect("equal"); ax.axis("off")
        ax.set_title(f"{label}\ndistance {d[i]:.3f}, material {m[i]:.2f}, {len(mechs[i]['x0'])} joints", fontsize=10)
        fig.colorbar(sc, ax=ax, fraction=0.04, pad=0.02, label="error at this target point")

        ax = axs[row, 1]
        order = np.argsort(-err[i])
        worst = np.zeros(lc.T, dtype=bool)
        worst[order[: lc.T // 5]] = True
        share = float(err[i][worst].sum() / err[i].sum())
        ax.fill_between(pos, 0, err[i], color="#f0c9a8", lw=0)
        ax.fill_between(pos, 0, np.where(worst, err[i], 0), color="#c8601a", lw=0, step="mid")
        ax.plot(pos, err[i], color="#16202e", lw=1)
        ax.set_xlim(0, 100); ax.set_ylim(0)
        ax.set_xlabel("Position along the target (% of perimeter, matching the circled marks)")
        ax.set_ylabel("Error")
        ax.set_title(f"The worst fifth of the perimeter (dark) carries {share:.0%} of the distance", fontsize=10)
        rows.append({"pick": label, "distance": float(d[i]), "material": float(m[i]), "joints": len(mechs[i]["x0"]),
                     "share_of_error_in_worst_fifth": share,
                     "worst_positions_percent": sorted(int(p) for p in pos[order[:10]])})
    fig.suptitle(f"{NAMES[t]}: where the distance comes from", fontsize=14)
    fig.tight_layout()
    fig.savefig(f"{OUT}/error_{t + 1}.png", dpi=120)
    plt.close(fig)

    # the family: average profile of the most accurate designs
    top = np.argsort(d)[:50]
    mean_err = err[top].mean(0)
    fig, axs = plt.subplots(1, 2, figsize=(13, 4.8), gridspec_kw={"width_ratios": [1, 1.5]})
    sc = axs[0].scatter(tgt[top[0], :, 0], tgt[top[0], :, 1], c=mean_err, cmap="YlOrRd", vmin=0, s=26,
                        edgecolors="#8a8a8a", linewidths=0.3)
    perimeter_marks(axs[0], tgt[top[0]] * 1.12)
    axs[0].set_aspect("equal"); axs[0].axis("off")
    fig.colorbar(sc, ax=axs[0], fraction=0.04, pad=0.02, label="mean error")
    axs[1].fill_between(pos, np.percentile(err[top], 10, axis=0), np.percentile(err[top], 90, axis=0),
                        color="#cdd9ea", lw=0, label="10th to 90th percentile")
    axs[1].plot(pos, mean_err, color="#1f5fa8", lw=1.6, label="mean")
    axs[1].set_xlim(0, 100); axs[1].set_ylim(0)
    axs[1].set_xlabel("Position along the target (% of perimeter)"); axs[1].set_ylabel("Error")
    axs[1].legend(fontsize=9)
    fig.suptitle(f"{NAMES[t]}: error shared by the {len(top)} most accurate designs "
                 f"(distance {d[top].min():.3f} to {d[top].max():.3f})", fontsize=13)
    fig.tight_layout()
    fig.savefig(f"{OUT}/family_{t + 1}.png", dpi=120)
    plt.close(fig)

    order = np.argsort(-mean_err)
    summary["targets"].append({
        "name": NAMES[t], "designs": len(mechs), "picks": rows,
        "family_share_of_error_in_worst_fifth": float(mean_err[order[: lc.T // 5]].sum() / mean_err.sum()),
        "family_worst_positions_percent": sorted(int(p) for p in pos[order[:20]]),
        "family_mean_error_by_tenth_of_perimeter": [float(mean_err[i:i + 20].mean()) for i in range(0, lc.T, 20)],
    })

with open(f"{OUT}/summary.json", "w") as f:
    json.dump(summary, f, indent=1)
print(json.dumps(summary, indent=1))
latest = "results/diagnose/latest"
shutil.rmtree(latest, ignore_errors=True)
shutil.copytree(OUT, latest)
print(f"saved {OUT} (also copied to {latest})")
