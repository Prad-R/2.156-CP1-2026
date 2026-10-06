# 2.156 CP1 — Linkage Synthesis

Class assignment (MIT 2.156, Fall 2026). Design planar linkages whose traced curve matches three kangaroo outlines while minimizing total link length. Course repo: `decode-mit/2.156-CP1-2026`.

## Standing instructions

These apply to whoever is working in this repo, each on their own branch. The `gh` login noted below is Prad's; `slides/` is local to each checkout.

- **Compute:** run everything as Slurm jobs, preempt-safe, checkpointed and requeuable; never run optimizations on the login node. Prefer `mit_preemptable`; when its queue is backed up, `mit_normal` or `mit_normal_gpu` may be used as well (user, 2026-10-05), under the same preempt-safe rules. Submit with `sbatch -p mit_preemptable,mit_normal -J <name> slurm/run.sbatch <script.py> [args]` so Slurm picks whichever starts first; never submit duplicates of a job, they would share a checkpoint.
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
- CPU-only jobs are accepted; max walltime is 2 days on `mit_preemptable`, 12 h on `mit_normal` (96 CPUs per user), 6 h on `mit_normal_gpu`.
- The batch script is copied at submit time: after editing `slurm/run.sbatch`, pending jobs must be cancelled and resubmitted.
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
- `refine.py` — gradient refinement (Adam, distance + w x material, four weights per seed) of screen seeds, one array task per target; checkpoint `checkpoints/refine/target_T.pkl` holds members (`mechs`, `last_good`), `front` (`F`, `x`, `src`). Publishes via `merge_and_publish`.
- `grow.py` — growth stage: add one dyad (rigid / dyad / new-ground) to each parent, prune, resize, refine best children, select parents per joint-count bin. Starts from the refine checkpoint; own checkpoint in `checkpoints/grow/`. Publishes each generation.
- `report_best.py` — figures and summary of the best submission and stage progress into `results/best/<timestamp>/` and `latest/`; the deck's optimization slides read `results/best/latest/`.
- `refine_wide.py` — successive-halving refinement from about 4,000 candidates per target (screen seeds and fronts, round one, growth, other targets' best); checkpoint `checkpoints/wide/target_T.pkl` (`run` = current stage arrays, `front` with `mechs`).
- `linkcore.py` — shared packing, kernels (`score`, `align`, `value_and_grad`, `traced_radius`), front helpers, parent pickers and the batched Adam update; newer scripts import it.
- `diagnose.py` — per-point error along each target for the best designs; figures in `results/diagnose/latest/`.
- `flip.py`, `reshape.py`, `polish.py` — experiments that all came back negative (see Things learned); kept for the record.
- `publish_best.py` — per-problem best submission, auto-commit and push (works from compute nodes).
- `advanced_starter.py` — script version of the advanced notebook (baseline only).
- `slurm/` — `run.sbatch`, `ckpt.py`, `smoke_test.py`.

## Things learned

- Pruning a mechanism to the ancestors of the traced joint removes dead material; the traced joint becomes the last joint.
- Distance is not scale-normalized, and uniform scaling of `x0` scales curve and material together, so scale is a free one-variable trade-off per mechanism.
- JAX here runs in float32. `Tools` re-jits per batch size: pad batches to a fixed size.
- Good designs so far are small (4 to 9 joints).

- Gradients on padding joints are nan by construction (zero-length links); mask them before testing a member for failure.
- The solver never solves a moving joint at index 2 (it assumes a second ground there); such designs come back constant and score as invalid.
- The screen's seed pool has no four-joint designs (they are less accurate), yet four-joint designs hold the low-material end of the fronts: include front designs in any later refinement round.
- Moving joint positions is exhausted (2026-10-06): wide restarts from about 4,000 designs, assembly-branch flips, chamfer-then-official and blurred-target losses, and strictly-improving polish all left best distances unchanged (about 0.052 / 0.22 / 0.45). Only structural change (growth) still lowers distance. Do not retry these without a new idea.
- Restarting Adam at lr 0.004 x size on converged designs usually leaves them worse; harvest the best point along the way, never the final state.
- Per-point error: Kangaroo 3 loses most on the tail (55 to 65% of perimeter); Kangaroo 2's best has four similar peaks; cheap Kangaroo 2 designs miss the snout-to-feet concavity.
- Score value is distance gain times material headroom, so accuracy gains on cheap designs are worth more than at the material-heavy tip of the front.

## Results so far

- 2026-10-05 baseline (job 24959498, Kangaroo 2 only): seeded GA hypervolume 0.204, after gradient step 0.408; scored submission 0.045 overall.
- 2026-10-05 16:13 mass screening after about 1 worker-hour (workers 24964610, merge 24964619): official overall score 2.61; hypervolume 4.14 / 6.16 / 16.62, normalized 2.07 / 4.11 / 1.66; best distance 0.19 / 0.51 / 0.87. Workers budgeted 4 h each.
- 2026-10-05 16:23 second merge (job 24967031, 2.7M mechanisms): overall 2.63; hypervolume 4.26 / 6.16 / 16.65. Gains from random screening are flattening.
- 2026-10-05 16:35 merge (job 24967804): overall 2.68; hypervolume 4.48 / 6.18 / 16.70. Kangaroo 3 front includes three bare-crank (2-joint) designs; user has been told, no decision yet on keeping them.
- 2026-10-05 ~16:55: refinement test (60 steps, Kangaroo 2) took overall to 2.86; one small growth generation on Kangaroo 3 took it to 2.93. Full refinement = job array 24969336; growth = 24969663 (starts after refinement). Deadline per user: about 1.5 days from 2026-10-05 afternoon.
- 2026-10-05 17:56: refinement round one done, overall 3.43 (hypervolume 5.50 / 7.92 / 22.74; best distance 0.067 / 0.30 / 0.48). Growth (24969663) running. 
- 2026-10-05 20:33: growth round one done, overall 3.50 (hypervolume 5.52 / 8.16 / 23.07; best distance 0.054 / 0.22 / 0.46). Wide refinement = job array 24993928 (`checkpoints/wide`), growth round two = 24993929 (`checkpoints/grow2`, starts after wide). User decided to keep bare-crank designs. User finds Kangaroo 2 corners and Kangaroo 3 tail/limbs not caught; told that the metric matches by arc-length position so partial features earn little.
- Queue note: in the evening short jobs waited 30+ minutes (reason Priority); do not rely on quick test jobs then.
- 2026-10-06 00:21: overall 3.69 (hypervolume 5.87 / 8.52 / 24.54). Growth round two (24994402, `checkpoints/grow2`) still running. Slides rebuilt at this point (22 slides).
- Slides: user asked (2026-10-05) for the deck to be rebuilt whenever there are new results, without being asked.
- 2026-10-06 afternoon: all stages finished, nothing running. Final score 3.6912 (hypervolume 5.865 / 8.521 / 24.604), verified from clean course code with `verify_submission.py`.
- Bare crank: the submission website says 4 to 20 joints per mechanism; Problem 3 has one 2-joint design. The user was told and chose to keep it ("restore the crank", 2026-10-06). Do not remove it or add joint-count enforcement unless the user asks.
- Hand-in: `CODE.md` and `2.156_CP1_code.zip` (rebuild command at the bottom of `CODE.md`; rebuild after any code or submission change).
- An interrupted Bash command may already have run most of its steps, including jobs that push: check the actual state before reporting it as not run.

## Open questions

- No report requirements or due date found in the repo; ask the user for the handout if it matters.
