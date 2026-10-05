# 2.156 CP1 — Linkage Synthesis

Class assignment (MIT 2.156, Fall 2026). Design planar linkages whose traced curve matches three kangaroo outlines while minimizing total link length. Course repo: `decode-mit/2.156-CP1-2026`.

## Standing instructions

These apply to whoever is working in this repo, each on their own branch. The `gh` login noted below is Prad's; `slides/` is local to each checkout.

- **Compute:** run everything as Slurm jobs on `mit_preemptable` only, preempt-safe and requeuable. Never run optimizations on the login node. Submit with `sbatch -J <name> slurm/run.sbatch <script.py> [args]`.
- **Git:** each person works only on their own branch, and commits and pushes there after each change. Prad's is `prads-branch`; a teammate uses the branch they created. Check `git branch --show-current` and who you are working with before committing, and never touch someone else's branch. `main` stays identical to the course repo.
- **Team:** two teammates work independently on separate branches; at the end they will pick one approach.
- **`UPDATES.md`:** add an entry for every change made (newest first, dated, with commit hash and job IDs). It is the user's change log.
- **Slide deck:** whenever there is a significant result, add slides so the user can follow what is happening. The deck is local and git-ignored (user's choice, 2026-10-05): edit `slides/build_deck.py` (append `slide(...)` calls before the final status slide) and rebuild with `~/.conda/envs/2.156_slides/bin/python slides/build_deck.py`, which writes `slides/deck.html` and `slides/deck.pdf` (the user wants the PDF). Screening numbers are read from `results/screen/latest/summary.json`. Layout must stay table/block based: WeasyPrint mis-renders nested flex.
- **Best submission:** `submissions/best_submission.npy` must always be the best found so far and be on GitHub for the team. `publish_best.publish(submission, score, source)` does this (per-problem best, re-score, commit, push); `merge_screen.py` calls it. Any new pipeline that produces a submission must call it too. These automated commits appear on the branch between manual ones.
- **This file:** keep it current as context for future sessions.

## Environment

- Conda env `2.156_slides` holds WeasyPrint and poppler for the PDF deck only.
- Conda env `2.156_cp_1` (`module load miniforge/26.7.2-0`; python at `~/.conda/envs/2.156_cp_1/bin/python`). Python 3.10, jax 0.5.3, pymoo 0.6.1, numpy 2.0.0, `gh` (logged in as `Prad-R`).
- `conda run` does not forward stdin; call the env's python directly for heredocs.
- Remotes: `origin` = `git@github.com:Prad-R/2.156-CP1-2026.git` (private); `upstream` = course repo, push disabled.
- Shared HPC login node (MIT ORCD Engaging): keep commands light and scoped; see `/etc/claude-code/CLAUDE.md`.

## Slurm facts

- `mit_preemptable` has `GraceTime=0` and `PreemptMode=REQUEUE`: a preempted job is killed with no signal and requeued. Scripts must checkpoint periodically (`slurm/ckpt.py`: `save`, `load`, `Every`) and resume on start. A signal handler alone is not enough.
- `slurm/run.sbatch` defaults: 4 CPUs, 8 GB, 4 h; override on the `sbatch` command line. Logs append to `logs/<name>-<jobid>.out`. It requeues itself 120 s before the time limit.
- CPU-only jobs are accepted; max walltime is 2 days.
- `logs/`, `checkpoints/`, `outputs/` are git-ignored.

## The problem

- Mechanism = `x0` (N×2 joint positions), `edges` (E×2), `fixed_joints`, `motor` (2,), `target_joint`.
- Two objectives, both minimized: distance (curve mismatch after optimal rotation/ordering, scale not normalized) and material (total link length).
- Per-target limits double as the validity cutoff (strictly under both) and the hypervolume reference point. Max 20 joints, max 1000 designs per target.

| Problem | Target | Distance limit | Material limit | Normalizer |
|---|---|---|---|---|
| 1 | Kangaroo 1 (round body) | 0.75 | 10 | 2.0 |
| 2 | Kangaroo 2 (no ears/tail) | 1.2 | 10 | 1.5 |
| 3 | Kangaroo 3 (full meme) | 1.75 | 20 | 10.0 |

- Score = mean over problems of hypervolume / normalizer. A missing or empty problem scores 0.
- Submission: one `.npy` dict with keys `Problem 1..3`, each a list of mechanism dicts. Score locally with `LINKS.CP.evaluate_submission`.

## Code map

- `LINKS/` — course library: `Optimization` (`Tools`, `DifferentiableTools`, `MechanismRandomizer`), `Kinematics.MechanismSolver`, `Geometry.CurveEngine`, `Visualization`, `CP` (limits, normalizers, scoring).
- `screen.py` — mass random sampling and screening for all three targets (Slurm array, one checkpoint per worker in `checkpoints/screen/worker_N.pkl`, readable summary in `worker_N.json`). Each checkpoint holds per-target Pareto `fronts` and the 2000 most accurate `seeds` (pruned mechanisms, already scaled). Uses its own skeleton generator: the course `MechanismRandomizer._skeleton_only` recurses without bound at higher ground probabilities.
- `merge_screen.py` — merges worker checkpoints (read-only, safe while workers run) and scores with `evaluate_submission`. Each merge goes to `results/screen/<timestamp>/` (`submission.npy`, `summary.json`, `fronts.png`, `best_<k>.png` drawings of the best mechanisms per target); `results/screen/latest/` mirrors the newest. Commit each merge folder.
- `publish_best.py` — per-problem best submission, auto-commit and push (works from compute nodes).
- `advanced_starter.py` — script version of the advanced notebook (baseline only).
- `slurm/` — `run.sbatch`, `ckpt.py`, `smoke_test.py`.

## Things learned

- Pruning a mechanism to the ancestors of the traced joint removes dead material; the traced joint becomes the last joint.
- Distance is not scale-normalized, and uniform scaling of `x0` scales curve and material together, so scale is a free one-variable trade-off per mechanism.
- JAX here runs in float32. `Tools` re-jits per batch size: pad batches to a fixed size.
- Good designs so far are small (4 to 9 joints).

## Results so far

- 2026-10-05 baseline (job 24959498, Kangaroo 2 only): seeded GA hypervolume 0.204, after gradient step 0.408; scored submission 0.045 overall.
- 2026-10-05 16:13 mass screening after about 1 worker-hour (workers 24964610, merge 24964619): official overall score 2.61; hypervolume 4.14 / 6.16 / 16.62, normalized 2.07 / 4.11 / 1.66; best distance 0.19 / 0.51 / 0.87. Workers budgeted 4 h each.
- 2026-10-05 16:23 second merge (job 24967031, 2.7M mechanisms): overall 2.63; hypervolume 4.26 / 6.16 / 16.65. Gains from random screening are flattening.
- 2026-10-05 16:35 merge (job 24967804): overall 2.68; hypervolume 4.48 / 6.18 / 16.70. Kangaroo 3 front includes three bare-crank (2-joint) designs; user has been told, no decision yet on keeping them.

## Open questions

- No report requirements or due date found in the repo; ask the user for the handout if it matters.
