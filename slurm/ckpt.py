"""Atomic checkpoint helpers for preemptable jobs.

A preempted job on mit_preemptable is killed without warning, so state must be
written often and written atomically (a half-written file must never replace a
good one).
"""
import os
import pickle
import time


def save(path, state):
    """Write `state` to `path` atomically."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = f"{path}.tmp.{os.getpid()}"
    with open(tmp, "wb") as f:
        pickle.dump(state, f, protocol=pickle.HIGHEST_PROTOCOL)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def load(path, default=None):
    """Return the checkpointed state, or `default` if there is none yet."""
    if not os.path.exists(path):
        return default
    with open(path, "rb") as f:
        return pickle.load(f)


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
