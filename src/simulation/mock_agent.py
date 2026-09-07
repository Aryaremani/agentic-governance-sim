"""A non-LLM agent for demoing the simulation instantly, with no GPU/Ollama required.

Strategies:
- "cooperative": stays near the sustainable cap, regardless of enforcement.
- "greedy": extracts near max_catch whenever enforcement is off, drops to the cap
  (to dodge fines) whenever enforcement is on.
- "random": extracts a random amount each round.

This lets anyone clone the repo and run a full experiment with `--mock` in seconds,
which matters a lot for a portfolio project — reviewers shouldn't need a GPU to see
it work.
"""

import random
from dataclasses import dataclass


@dataclass
class MockAgent:
    agent_id: str
    strategy: str = "random"  # cooperative | greedy | random
    model_name: str = "mock"  # kept for interface parity with Agent

    def decide_extraction(
        self,
        n_agents: int,
        max_catch: int,
        sustainable_cap: int,
        enforcement_active: bool,
        recent_context: str = "",
    ) -> float:
        if self.strategy == "cooperative":
            return sustainable_cap + random.uniform(-0.5, 0.5)
        if self.strategy == "greedy":
            if enforcement_active:
                return sustainable_cap + random.uniform(0, 0.5)  # just dodges the fine
            return max_catch * random.uniform(0.85, 1.0)
        # random
        return random.uniform(0, max_catch)
