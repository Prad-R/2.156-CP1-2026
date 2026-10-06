#!/bin/bash
# Preempt-safe, requeuable job wrapper for 2.156 CP1.
#
#   sbatch -J <name> [-t HH:MM:SS] [-c N] slurm/run.sbatch <script.py> [args...]
#
# Default partition is mit_preemptable. When its queue is backed up, submit with
#   -p mit_preemptable,mit_normal
# and Slurm runs the job on whichever starts first (mit_normal: 12 h limit,
# 96 CPUs per user; mit_normal_gpu: 6 h limit). Jobs must be preempt-safe on
# every partition all the same.
#
# mit_preemptable has GraceTime=0: a preempted job is killed with no warning and
# put back in the queue. So the python script must checkpoint periodically and
# resume from its checkpoint on start (see slurm/ckpt.py) -- a signal handler
# alone is not enough. Logs append across restarts.

#SBATCH --partition=mit_preemptable
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=04:00:00
#SBATCH --signal=B:USR1@120
#SBATCH --output=logs/%x-%j.out

set -uo pipefail

case "${SLURM_JOB_PARTITION:-}" in
    mit_preemptable|mit_normal|mit_normal_gpu) ;;
    *) echo "Refusing to run on partition '${SLURM_JOB_PARTITION:-none}'" >&2; exit 1 ;;
esac

cd "${SLURM_SUBMIT_DIR}"

module load miniforge/26.7.2-0
eval "$(conda shell.bash hook)"
conda activate 2.156_cp_1

export JAX_PLATFORMS=cpu
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export XLA_FLAGS="--xla_force_host_platform_device_count=1"
export PYTHONUNBUFFERED=1

echo "=== $(date '+%F %T') job ${SLURM_JOB_ID} on $(hostname), restart #${SLURM_RESTART_COUNT:-0}: $* ==="

# Time limit approaching (USR1 arrives 120s early): requeue so the run resumes
# from its last checkpoint instead of dying unfinished.
requeue_on_timeout() {
    echo "=== $(date '+%F %T') time limit near, requeueing ==="
    kill -TERM "${PY_PID}" 2>/dev/null
    wait "${PY_PID}" 2>/dev/null
    scontrol requeue "${SLURM_JOB_ID}"
    exit 0
}
trap requeue_on_timeout USR1

python "$@" &
PY_PID=$!
wait "${PY_PID}"
RC=$?
echo "=== $(date '+%F %T') python exited with ${RC} ==="
exit "${RC}"
