# Updates

A running log of what has changed in this repo and why, newest first. Results worth understanding also go into the slide deck.

## 2026-10-05

### Project tracking files
- Added `CLAUDE.md` (working context for Claude) and this file.
- Started a results slide deck; link is in `CLAUDE.md`.

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
