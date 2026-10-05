"""Smoke test: checks the env on a compute node and that checkpoint/resume works.

Random-perturbs the starter mechanism for a fixed number of steps, keeping the
best distance on Kangaroo 1. Resumes from checkpoints/smoke.pkl if it exists.
"""
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from slurm import ckpt
from LINKS.Optimization import Tools

CKPT = "checkpoints/smoke.pkl"
N_STEPS = 300

target = np.load("kangaroo_target_curves.npy")[0]
mech = np.load("starter_mechanism.npy", allow_pickle=True).item()
tools = Tools(device="cpu")
tools.compile()


def evaluate(x0):
    return tools(x0, mech["edges"], mech["fixed_joints"], mech["motor"], target, target_idx=None)


state = ckpt.load(CKPT)
if state is None:
    d, m = evaluate(mech["x0"])
    state = {"step": 0, "best_x0": mech["x0"], "best": (float(d), float(m)),
             "rng": np.random.default_rng(0).bit_generator.state}
    print(f"fresh start: distance={d:.4f} material={m:.4f}")
else:
    print(f"RESUMED from step {state['step']}: best distance={state['best'][0]:.4f}")

rng = np.random.default_rng()
rng.bit_generator.state = state["rng"]

while state["step"] < N_STEPS:
    cand = state["best_x0"] + rng.normal(scale=0.01, size=state["best_x0"].shape)
    d, m = evaluate(cand)
    if np.isfinite(d) and d < state["best"][0]:
        state["best_x0"], state["best"] = cand, (float(d), float(m))
    state["step"] += 1
    state["rng"] = rng.bit_generator.state
    if state["step"] % 20 == 0:
        ckpt.save(CKPT, state)
        print(f"step {state['step']}: best distance={state['best'][0]:.4f} material={state['best'][1]:.4f}")
    time.sleep(0.2)

ckpt.save(CKPT, state)
print(f"DONE: distance={state['best'][0]:.4f} material={state['best'][1]:.4f}")
