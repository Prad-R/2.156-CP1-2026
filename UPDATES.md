# Updates

A running log of what has changed in this repo and why, newest first. Results worth understanding also go into the slide deck.

## 2026-10-05

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
