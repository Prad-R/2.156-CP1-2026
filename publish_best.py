"""Keep submissions/best_submission.npy as the best submission found so far.

The three problems are scored independently, so "best" is tracked per problem:
a new submission replaces a problem's designs only where its hypervolume for
that problem is higher. When anything improves, the file is re-scored with
evaluate_submission, committed and pushed, so teammates can always fetch it:

    git fetch origin
    git checkout origin/prads-branch -- submissions/

Called automatically at the end of merge_screen.py; any pipeline can call
`publish(submission, score, source)`.
"""
import json
import os
import subprocess

import numpy as np

from LINKS.CP import evaluate_submission

DIR = "submissions"
NPY = f"{DIR}/best_submission.npy"
META = f"{DIR}/best_submission.json"
PROBLEMS = ["Problem 1", "Problem 2", "Problem 3"]


def _git(*args):
    r = subprocess.run(["git", *args], capture_output=True, text=True, timeout=120)
    return r.returncode, (r.stdout + r.stderr).strip()


def push():
    """Commit and push the best-submission files; safe to call when nothing changed."""
    _git("add", NPY, META)
    code, _ = _git("diff", "--cached", "--quiet", "--", NPY, META)
    if code != 0:
        with open(META) as f:
            meta = json.load(f)
        msg = (f"Update best submission: overall {meta['score']['Overall Score']:.4f}\n\n"
               f"Per-problem sources: {json.dumps(meta['source'])}\n\n"
               "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>")
        code, out = _git("commit", "--only", NPY, META, "-m", msg)
        if code != 0:
            print(f"best submission: commit failed: {out}")
            return False
    code, out = _git("push", "origin", "HEAD")
    print("best submission: pushed to GitHub" if code == 0
          else f"best submission: push failed (file is committed locally; push later): {out}")
    return code == 0


def publish(submission, score, source):
    """Fold `submission` into the best file wherever it scores higher; push if changed."""
    os.makedirs(DIR, exist_ok=True)
    if os.path.exists(NPY) and os.path.exists(META):
        best = np.load(NPY, allow_pickle=True).item()
        with open(META) as f:
            meta = json.load(f)
    else:
        best = {k: [] for k in PROBLEMS}
        meta = {"hypervolume": {k: 0.0 for k in PROBLEMS}, "source": {k: None for k in PROBLEMS}}

    improved = []
    for k in PROBLEMS:
        hv = score["Score Breakdown"][k]
        if hv > meta["hypervolume"][k] + 1e-9:
            best[k] = submission[k]
            meta["hypervolume"][k], meta["source"][k] = hv, source
            improved.append(k)
    if not improved:
        print("best submission: no problem improved, nothing to publish")
        push()   # still push in case an earlier push failed
        return False

    meta["score"] = evaluate_submission(best)
    meta["hypervolume"] = dict(meta["score"]["Score Breakdown"])
    meta["designs"] = {k: len(best[k]) for k in PROBLEMS}
    tmp = NPY + ".tmp.npy"
    np.save(tmp, best)
    os.replace(tmp, NPY)
    with open(META, "w") as f:
        json.dump(meta, f, indent=1)
    print(f"best submission: improved {improved}, overall {meta['score']['Overall Score']:.4f}")
    push()
    return True
