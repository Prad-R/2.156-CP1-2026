"""Score the best submission the way a grader would, from a clean copy of the course code.

Extracts the course repo's main branch (upstream/main) into a scratch folder,
then in a fresh Python process there loads the submission file by path and
runs the course's own evaluate_submission on it. Nothing from this branch's
code is imported. The result is compared with the score we recorded.

    sbatch -p mit_preemptable,mit_normal -J verify -c 4 --mem=8G -t 00:15:00 slurm/run.sbatch verify_submission.py
"""
import json
import os
import shutil
import subprocess
import sys

SUBMISSION = os.path.abspath(sys.argv[1] if len(sys.argv) > 1 else "submissions/best_submission.npy")
CLEAN = os.path.abspath("checkpoints/verify_clean")

shutil.rmtree(CLEAN, ignore_errors=True)
os.makedirs(CLEAN)
archive = subprocess.run(["git", "archive", "upstream/main"], check=True, capture_output=True).stdout
subprocess.run(["tar", "-x", "-C", CLEAN], input=archive, check=True)
head = subprocess.run(["git", "rev-parse", "--short", "upstream/main"], capture_output=True, text=True).stdout.strip()

code = (
    "import json, numpy as np\n"
    "from LINKS.CP import evaluate_submission\n"
    f"sub = np.load({SUBMISSION!r}, allow_pickle=True).item()\n"
    "print('LOADED', type(sub).__name__, {k: len(v) for k, v in sub.items()})\n"
    f"print('SCORE', json.dumps(evaluate_submission({SUBMISSION!r}, 'kangaroo_target_curves.npy')))\n"
)
env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
run = subprocess.run([sys.executable, "-c", code], cwd=CLEAN, env=env, capture_output=True, text=True)
print(run.stdout)
if run.returncode != 0:
    print(run.stderr[-3000:])
    raise SystemExit("clean-room scoring FAILED")

warnings = [line for line in run.stdout.splitlines() if "Warning" in line]
clean = json.loads(next(line for line in run.stdout.splitlines() if line.startswith("SCORE"))[6:])
with open("submissions/best_submission.json") as f:
    recorded = json.load(f)["score"]
print(f"course code at upstream/main {head}")
print(f"clean-room overall score : {clean['Overall Score']:.6f}")
print(f"recorded overall score   : {recorded['Overall Score']:.6f}")
for k in clean["Score Breakdown"]:
    print(f"  {k}: clean {clean['Score Breakdown'][k]:.6f}  recorded {recorded['Score Breakdown'][k]:.6f}")
print("scorer warnings:", warnings or "none")
match = abs(clean["Overall Score"] - recorded["Overall Score"]) < 1e-3
print("VERDICT:", "MATCH" if match and not warnings else "CHECK NEEDED")
