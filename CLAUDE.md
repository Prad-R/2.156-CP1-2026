# 2.156 CP1 — Linkage Synthesis

Class assignment (MIT 2.156, Fall 2026). Design planar linkages whose traced curve matches three kangaroo outlines while minimizing total link length. Course repo: `decode-mit/2.156-CP1-2026`.

## Standing instructions from the user

- **Compute:** run everything as Slurm jobs on `mit_preemptable` only, preempt-safe and requeuable. Never run optimizations on the login node. Submit with `sbatch -J <name> slurm/run.sbatch <script.py> [args]`.
- **Git:** make changes on `prads-branch` only, and commit and push there after each change (user said "commit and push" on 2026-10-05 when asked about a standing go-ahead). `main` stays identical to the course repo.
- **Team:** the user's teammate will work on a separate branch of their own; at the end the two will pick one approach. Do not touch other branches.
- **`UPDATES.md`:** add an entry for every change made (newest first, dated, with commit hash and job IDs). It is the user's change log.
- **Slide deck:** whenever there is a significant result, add slides to the results deck so the user can follow what is happening: https://claude.ai/artifact/TsoK36ZuYXQw6jXmsN7vHj (Slides artifact; read `project/deck.json` there before revising, append new result slides before the `status` slide and update `status`).
- **This file:** keep it current as context for future sessions.

## Environment

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
- `advanced_starter.py` — script version of the advanced notebook (mixed-variable NSGA-II seeded from random mechanisms, then gradient descent on distance). Target chosen by `target_index` (default 1 = Kangaroo 2). Figures go to `outputs/advanced/`. Not checkpointed; runs in about 2 minutes.
- `slurm/` — `run.sbatch`, `ckpt.py`, `smoke_test.py`.

## Results so far

- 2026-10-05 baseline (job 24959498, Kangaroo 2 only): plain GA found nothing feasible; seeded GA hypervolume 0.204; after gradient step 0.408. Scored submission 0.045 overall (built before the gradient step; Problems 1 and 3 empty).

## Open questions

- No report requirements or due date found in the repo; ask the user for the handout if it matters.
