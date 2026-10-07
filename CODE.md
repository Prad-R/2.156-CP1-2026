# Code guide

What each code file on this branch is, what it was used for, and whether it contributed to the final submission. The same files are packaged in `2.156_CP1_code.zip`.

## The pipeline, in the order it ran

| Step | File | What it does | Role in the final submission |
|---|---|---|---|
| 0 | `advanced_starter.py` | The course's advanced notebook converted to a script (mixed-variable NSGA-II seeded with random mechanisms, then gradient descent on distance). Run unchanged. | Baseline only: score 0.045. None of its designs are in the submission. |
| 1 | `screen.py` | Mass random search for all three targets. Samples random dyadic mechanisms, treats every moving joint as a candidate tracer, prunes each candidate to the links that drive it, pre-filters by shape with an FFT alignment, then scores the best with the official metric at several sizes. | Produced the starting designs for everything after it. Screening alone scored 2.75. |
| 1 | `merge_screen.py` | Merges the screening workers' checkpoints into one scored submission, with figures of the fronts and best mechanisms. | Bookkeeping and figures. |
| 2 | `refine.py` | Gradient refinement (Adam on joint positions) of the most accurate screened designs per joint count, each at four material weights, minimizing distance + weight x material. | Raised the score to 3.43. |
| 3 | `grow.py` | Growth: attaches one new traced joint with two links to each good design in many ways, keeps and refines the best children, and selects the next parents per joint count, inside the material limit. Run twice. | Raised the score to 3.50 (round one). Round two made the last improvement to all three problems in the final file. |
| 4 | `refine_wide.py` | Refinement from about 4,000 starting designs per target with successive halving: short run for all, longer for the best quarter, long run at several material weights for the finalists. | Raised the score to 3.67, through cheaper designs at the same accuracy. |
| 5 | `polish.py` | Takes designs from along the front and accepts only steps that improve one objective without worsening the other. | Negligible gain (under 0.002 hypervolume per target). Showed the front designs sit at true minima. |

Final official score: 3.69 overall.

## Experiments that did not improve the result

Kept for the record; each was a controlled test of an idea for lowering distance.

| File | Idea tested | Outcome |
|---|---|---|
| `flip.py` | Put each moving joint on the other side of the line through its parents (the other assembly branch), keep all link lengths, then refine. | About 7,000 flips tried. Hypervolume changed by 0.012 or less. |
| `reshape.py` | Refine with a more forgiving objective first (a two-way nearest-point distance, or the official distance against a blurred target sharpened in steps), then finish on the official distance. Compared against a control on the same parents. | The control matched or beat both alternatives on every target. |

## Analysis, reporting and checking

| File | What it does |
|---|---|
| `diagnose.py` | Aligns designs exactly as the scorer does and plots the error point by point along each target. Showed that Kangaroo 3 loses most of its distance on the tail and that Kangaroo 2 has four similar error peaks. |
| `report_best.py` | Figures and a summary for the current best submission: fronts, drawings of the best mechanisms, progress curves of the optimizing stages, and where the unclaimed score is. |
| `publish_best.py` | Keeps `submissions/best_submission.npy` as the best designs found so far, tracked per problem; re-scores with the course's `evaluate_submission`; commits and pushes when a problem improves. Every stage publishes through it. |
| `verify_submission.py` | Scores the submission file from a clean copy of the course repository in a fresh process, and compares with the recorded score. |

## Shared code and infrastructure

| File | What it does |
|---|---|
| `linkcore.py` | Shared by the later scripts: packing mechanisms into solver arrays, the jitted scoring, alignment and gradient kernels, Pareto-front helpers, parent selection, and the batched Adam update. |
| `slurm/run.sbatch` | Job wrapper. Runs any script as a requeueable Slurm job, appends logs across restarts, and requeues itself before the time limit. |
| `slurm/ckpt.py` | Atomic checkpoint save and load with a backup copy, so a job killed without warning resumes from its last minute of work. |
| `slurm/smoke_test.py` | Small job used to confirm that checkpoint and resume work after a forced requeue. |

## Course code, unmodified

| Path | What it is |
|---|---|
| `LINKS/` | The course library: kinematics solver, curve alignment and distance, optimization tools, visualization, and the scorer (`LINKS/CP`). Identical to the course repository. |
| `kangaroo_target_curves.npy`, `starter_mechanism.npy` | The three target curves and the starter mechanism from the course. |
| `requirements.txt` | The course's Python requirements. |

## Results

| Path | What it is |
|---|---|
| `submissions/best_submission.npy` | The submission file. |
| `submissions/best_submission.json` | Its official score, the number of designs per problem, and which run each problem's designs came from. |
| `report/report.pdf` | The project report. |
| `report/report.tex`, `report/figures/` | Its LaTeX source and figures (`pdflatex report.tex` twice inside `report/`). |

## Running it

Python 3.10 with the packages in `requirements.txt`. Every script runs from the repository root. The long ones are written for Slurm and resume from their checkpoints:

```bash
sbatch -J screen --array=0-3 -c 8 --mem=16G -t 08:00:00 -o logs/%x-%A_%a.out slurm/run.sbatch screen.py --hours 4
sbatch -J refine --array=0-2 -c 8 --mem=24G -t 08:00:00 -o logs/%x-%A_%a.out slurm/run.sbatch refine.py --steps 1500
sbatch -J grow   --array=0-2 -c 8 --mem=24G -t 12:00:00 -o logs/%x-%A_%a.out slurm/run.sbatch grow.py --gens 20
sbatch -J wide   --array=0-2 -c 8 --mem=12G -t 03:00:00 -o logs/%x-%A_%a.out slurm/run.sbatch refine_wide.py
sbatch -J grow2  --array=0-2 -c 8 --mem=12G -t 04:00:00 -o logs/%x-%A_%a.out slurm/run.sbatch grow.py --gens 20 --out checkpoints/grow2 --also-from checkpoints/wide checkpoints/grow
sbatch -J verify -c 4 --mem=8G -t 00:15:00 slurm/run.sbatch verify_submission.py
```

Each array task handles one target (or one screening worker). Without Slurm, the same scripts run directly, for example `python refine.py --target 1`.

The zip holds the code, the course library it needs, the submission file and the report. The change log (`UPDATES.md`) and other working files stay in the repository only. It is rebuilt from the committed files with:

```bash
git archive --format=zip --prefix=2.156_CP1_code/ -o 2.156_CP1_code.zip HEAD \
    CODE.md requirements.txt advanced_starter.py screen.py merge_screen.py refine.py refine_wide.py \
    grow.py polish.py flip.py reshape.py diagnose.py report_best.py publish_best.py verify_submission.py linkcore.py \
    slurm LINKS kangaroo_target_curves.npy starter_mechanism.npy \
    submissions/best_submission.npy submissions/best_submission.json report
```
