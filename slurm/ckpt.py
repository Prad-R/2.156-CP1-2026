"""Atomic checkpoint helpers for preemptable jobs.

A preempted job on mit_preemptable is killed without warning, so state must be
written often and written atomically (a half-written file must never replace a
good one). The previous good checkpoint is kept as `<path>.bak`.
"""
import os
import pickle
import time


def save(path, state):
    """Write `state` to `path` atomically, keeping the previous file as .bak."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = f"{path}.tmp.{os.getpid()}"
    with open(tmp, "wb") as f:
        pickle.dump(state, f, protocol=pickle.HIGHEST_PROTOCOL)
        f.flush()
        os.fsync(f.fileno())
    if os.path.exists(path):
        os.replace(path, path + ".bak")
    os.replace(tmp, path)


def load(path, default=None):
    """Return the checkpointed state, falling back to .bak, then `default`."""
    for p in (path, path + ".bak"):
        if not os.path.exists(p):
            continue
        try:
            with open(p, "rb") as f:
                return pickle.load(f)
        except Exception as e:  # truncated or corrupt file: try the backup
            print(f"checkpoint {p} unreadable ({e!r}), trying fallback")
    return default


class Every:
    """True at most once per `seconds`; use to rate-limit checkpoint writes."""

    def __init__(self, seconds):
        self.seconds = seconds
        self.last = time.monotonic()

    def __call__(self):
        now = time.monotonic()
        if now - self.last >= self.seconds:
            self.last = now
            return True
        return False
