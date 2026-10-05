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
import fcntl
import json
import os
import subprocess

import numpy as np

from LINKS.CP import REFERENCE_POINTS, evaluate_submission
from LINKS.Optimization import Tools

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
    with open(f"{DIR}/.lock", "w") as lock:          # several jobs may publish at once
        fcntl.flock(lock, fcntl.LOCK_EX)
        return _publish(submission, score, source)


def _publish(submission, score, source):
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


_tools = None


def _official(mechs, target, batch=512):
    """Distance and material from the scoring code, in fixed-size batches."""
    global _tools
    if _tools is None:
        _tools = Tools(timesteps=200, max_size=20, material=True, scaled=False, device="cpu")
        _tools.compile()
    out = []
    for i in range(0, len(mechs), batch):
        chunk = mechs[i:i + batch]
        pad = chunk + [chunk[0]] * (batch - len(chunk))
        d, m = _tools([np.array(c["x0"]) for c in pad], [np.array(c["edges"]) for c in pad],
                      [np.array(c["fixed_joints"]) for c in pad], [np.array(c["motor"]) for c in pad],
                      target, [c.get("target_joint") for c in pad])
        out.append(np.stack([np.asarray(d), np.asarray(m)], 1)[: len(chunk)])
    return np.concatenate(out)


def merge_and_publish(new, source, max_designs=1000):
    """Add designs to the best file. `new` maps 'Problem k' to a list of mechanisms.

    For each problem given, the new designs are pooled with the current best
    ones, scored with the official metric, reduced to the non-dominated set
    (at most `max_designs`, evenly spread) and published if that scores higher.
    """
    targets = np.load("kangaroo_target_curves.npy")
    current = np.load(NPY, allow_pickle=True).item() if os.path.exists(NPY) else {k: [] for k in PROBLEMS}
    submission = {k: list(v) for k, v in current.items()}
    for k, mechs in new.items():
        t = PROBLEMS.index(k)
        pool = list(current.get(k, [])) + list(mechs)
        if not pool:
            continue
        F = _official(pool, targets[t])
        ok = np.where(np.isfinite(F).all(1) & (F[:, 0] < REFERENCE_POINTS[t][0]) & (F[:, 1] < REFERENCE_POINTS[t][1]))[0]
        if not len(ok):
            continue
        order = ok[np.lexsort((F[ok, 1], F[ok, 0]))]
        keep = np.ones(len(order), dtype=bool)
        keep[1:] = F[order, 1][1:] < np.minimum.accumulate(F[order, 1])[:-1]
        idx = order[keep]
        if len(idx) > max_designs:
            idx = idx[np.linspace(0, len(idx) - 1, max_designs).round().astype(int)]
        submission[k] = [pool[i] for i in idx]
    return publish(submission, evaluate_submission(submission), source)
