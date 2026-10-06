# Updates

A running log of what has changed in this repo and why, newest first. Results worth understanding also go into the slide deck.

## 2026-10-05

### Jobs may now also run on mit_normal
- The preemptable queue backed up in the evening (about 1,170 jobs pending on priority; my jobs waited 30 to 60 minutes without starting). On the user's instruction, `slurm/run.sbatch` now also accepts `mit_normal` and `mit_normal_gpu`, and jobs are submitted with `-p mit_preemptable,mit_normal` so they start on whichever is free first. Same checkpoint and requeue rules everywhere.
- Resubmitted that way: wide refinement (24994401), growth round two (24994402, chained), report (24994403). The earlier queued copies were cancelled.
- `results/` is no longer tracked in git (commit `ea8556b`); figures and summaries stay on the cluster. Only `submissions/` holds generated files in the repo.

### Growth round one finished (overall 3.50); wide refinement and growth round two queued
- Growth (job array 24969663, 20 generations per target) finished between 19:21 and 20:33. Official score **3.50** overall; hypervolume 5.52 / 8.16 / 23.07. Best distance 0.054 / 0.22 / 0.46 (before growth: 0.067 / 0.30 / 0.48). The last generation found no improving children on any target.
- Flaw found: growth chose parents on distance alone, so some drifted over the material limit (Kangaroo 1 reached distance 0.040 with 20 joints, outside the limit, so it does not count). Fixed in `grow.py`: parents must be inside the limit and designs near it are pushed back.
- Added `refine_wide.py` (commits `a17f205`, `705a92e`): starts from about 4,000 candidates per target (all screening seeds and fronts, round one's designs, growth's designs, and the other targets' best), keeps the best per joint count after each stage, and gives finalists a long run at several material weights. Queued as job array 24993928.
- Growth round two (job array 24993929) is chained to start from the wide round's results.
- `report_best.py` now also reports where the unclaimed score is.
- Screening workers finished their 4-hour budgets at about 20:00.
- User decision: keep the bare-crank designs in the submission.
- Git: timestamped result folders and the publish lock are git-ignored; only `results/*/latest` and `submissions/` are tracked.
- The cluster queue slowed in the evening: short jobs waited 30 minutes or more, so the wide round was launched without its test run.

### Refinement round one finished: overall 3.43
- Refinement (job array 24969336, 1,500 steps per target) finished at 17:29 for Kangaroos 2 and 3 and 17:56 for Kangaroo 1. Official score from `evaluate_submission`: **3.43** overall (screening alone: 2.75).
  - Hypervolume 5.50 / 7.92 / 22.74; normalized 2.75 / 5.28 / 2.27.
  - Best distance 0.067 / 0.30 / 0.48 (screening: 0.17 / 0.50 / 0.81). Kangaroo 3's most accurate design now follows both ears.
- Growth stage (job array 24969663) started automatically at 17:56 on all three targets.
- Added `report_best.py`: figures and a summary for the current best submission plus progress curves of the optimizing stages, written to `results/best/<timestamp>/` and mirrored to `results/best/latest/`. Reports are scheduled through 23:27.
- Slide deck rebuilt (18 slides, HTML and PDF) with refinement results, progress curves, new fronts and best-mechanism drawings.

### Optimization stages added: gradient refinement and growth — commits `ef2cd00`, `3946b25`
- Up to here every design was a random mechanism that was only pruned and resized. Two optimizing stages now follow the screen.
- `refine.py`: Adam on joint positions for the most accurate seeds per joint count (5 to 9 and more), each at four material weights, minimizing distance + w x material. Running for all three targets as job array 24969336 (1,500 steps each).
  - 60-step test on Kangaroo 2: best distance 0.515 to 0.411, overall score 2.68 to 2.86.
  - First test showed every design as broken: my bug (gradients on unused padding slots counted as failures), fixed before launch.
- `grow.py`: adds one traced joint with two links to each good design, in many ways, keeps and refines the best children, and selects next parents per joint-count bin. Queued as job array 24969663 to start when refinement finishes (20 generations).
  - One small test generation on Kangaroo 3: 42 children beat the best parent, best distance 0.805 to 0.768, overall score to 2.93.
- `publish_best.py` gained `merge_and_publish` (pool new designs with the current best, re-score, publish if higher) and a lock so several jobs can publish at once. Both new stages publish automatically.
- Known gap: the refinement seeds include no four-joint designs, which hold the low-material end of the fronts. A second round should include the front designs.

### Best submission is published to GitHub automatically
- Added `publish_best.py`: keeps `submissions/best_submission.npy` as the best found so far, tracked per problem, with its score in `submissions/best_submission.json`. `merge_screen.py` calls it at the end of every merge; when any problem improves it re-scores, commits and pushes on its own.
- Tested with merge job 24967804 (16:35): pushed from the compute node as commit `49e1e81`, overall score **2.68** (hypervolume 4.48 / 6.18 / 16.70).
- Added `submissions/README.md` with fetch and load instructions for teammates.
- Note: the Kangaroo 3 front includes three two-joint designs (a bare crank tracing a circle). The course scorer accepts them.

### Slide deck as PDF
- `slides/build_deck.py` now also writes `slides/deck.pdf` (one page per slide) and reads the screening numbers from the latest merge. PDF export uses WeasyPrint in a separate env, `2.156_slides`.
- Best-mechanism figures are now landscape (one column per pick); a design that wins two picks is drawn once.

### Every merge now saves drawings of the best mechanisms
- `merge_screen.py` writes each merge to its own folder `results/screen/<timestamp>/` (nothing is overwritten) and mirrors the newest to `results/screen/latest/`.
- New `best_1.png` to `best_3.png`: for each target, the most accurate, best-balanced and least-material designs, drawn as mechanisms next to their traced curve on the target.
- Second merge (job 24967031, 16:23, 2.7 million mechanisms, 1.6 worker-hours): overall score **2.63**; hypervolume 4.26 / 6.16 / 16.65; normalized 2.13 / 4.11 / 1.66.
- The flat `results/screen_*` files from the first merge are replaced by this layout.

### Mass screening: official score 2.61 — commit `0f6cab2` and this one
- Added `screen.py`: samples random mechanisms, prunes each candidate joint's mechanism to the links that drive it, pre-filters by shape, then scores the best with the official metric at several sizes. Checkpoints atomically every minute; workers resume exactly where they stopped.
- Added `merge_screen.py`: merges all workers into one scored submission and figures in `results/`.
- `slurm/ckpt.py` now keeps the previous checkpoint as a `.bak` fallback.
- Launched 4 workers (job array 24964610, 4 h budget each). First merge (job 24964619) after about 1 worker-hour and 1.58 million mechanisms:
  - Overall score **2.61** (baseline 0.045), from `evaluate_submission`.
  - Hypervolume 4.14 / 6.16 / 16.62; normalized 2.07 / 4.11 / 1.66.
  - Best distance 0.19 / 0.51 / 0.87. All designs have 4 to 9 joints.
- Both the test job and the merge job were preempted mid-run and resumed correctly.
- Two failed test runs before launch: the course's random generator recursed without limit (replaced with my own), then a missing output folder (fixed).

### Slides moved local
- The deck is now `slides/deck.html`, built by `slides/build_deck.py`; `slides/` is git-ignored. The earlier claude.ai deck is gone.

### Project tracking files
- Added `CLAUDE.md` (working context for Claude) and this file.
- Started a results slide deck.

### Advanced notebook as a script — commit `359a570`
- Added `advanced_starter.py`, a cell-by-cell conversion of `Fall_26_CP1_Advanced_Starter_Notebook.ipynb`.
- Differences from the notebook: the Colab setup cell is removed, figures are saved to `outputs/advanced/` instead of shown, the submission score is printed, and progress bars are throttled.
- Ran it as Slurm job 24959498 (about 2 minutes, exit 0). Results on Kangaroo 2:
  - Plain GA from a random start, 6 joints: no feasible solution.
  - GA seeded with 100 random 7-joint mechanisms: hypervolume 0.204.
  - After gradient descent on distance: hypervolume 0.408.
  - Scored submission (built before the gradient step, Problem 2 only): overall 0.045.

### Slurm setup — commit `920997c`
- Added `slurm/run.sbatch`: runs any script on `mit_preemptable` with `--requeue`, refuses other partitions, appends logs across restarts, requeues itself before the time limit.
- Added `slurm/ckpt.py`: atomic checkpoint save/load.
- Added `slurm/smoke_test.py`: confirmed a job resumes from its checkpoint after a forced requeue (job 24958457).
- `logs/`, `checkpoints/` and `outputs/` are git-ignored.

### Environment and repo
- Cloned the course repo into `~/2.156/cp_1`.
- Created conda env `2.156_cp_1` (Python 3.10, jax 0.5.3, pymoo 0.6.1, numpy 2.0.0, `gh`).
- Created private GitHub repo `Prad-R/2.156-CP1-2026` as `origin`; the course repo is `upstream`, fetch-only.
- Work happens on `prads-branch`; `main` stays identical to the course repo.
