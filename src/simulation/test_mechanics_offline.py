"""Offline sanity check for ResourcePool + Enforcement logic, with random extraction
decisions standing in for real LLM agents. Run this before spending GPU time on real
Ollama calls, to confirm the environment mechanics behave as expected (regrowth,
depletion, fines, collapse detection).

Usage: python src/simulation/test_mechanics_offline.py
"""

import random
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from simulation.environment import ResourcePool
from simulation.enforcement import Enforcement


def run_offline(n_agents=5, n_rounds=40, greedy=False, seed=0):
    random.seed(seed)
    pool = ResourcePool(stock=80, capacity=100, regrowth_rate=0.20, sustainability_threshold=20)
    enforcement = Enforcement(sustainable_cap=8, fine_multiplier=1.5, active=False)

    for r in range(1, n_rounds + 1):
        # phase schedule matching configs/*.yaml
        if r <= 10:
            phase, enforcement.active = "no_enforcement", False
        elif r <= 25:
            phase, enforcement.active = "enforcement_active", True
        else:
            phase, enforcement.active = "enforcement_withdrawn", False

        requested = {}
        for i in range(n_agents):
            if greedy:
                requested[f"agent_{i+1}"] = 15 if not enforcement.active else 8
            else:
                requested[f"agent_{i+1}"] = random.uniform(4, 12)

        actual = pool.extract(requested)
        fines = enforcement.apply(actual)
        pool.log_round(r, phase, actual, fines)

        print(
            f"round {r:2d} [{phase:20s}] stock={pool.stock:6.2f} "
            f"extracted={sum(actual.values()):6.2f} collapsed={pool.is_collapsed()}"
        )

    return pool.history


if __name__ == "__main__":
    print("=== Random extraction ===")
    run_offline(greedy=False)
    print("\n=== Greedy extraction (max when unenforced) ===")
    run_offline(greedy=True)
