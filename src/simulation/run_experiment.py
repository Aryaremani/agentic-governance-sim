"""Run a full three-phase CPR experiment from a YAML config and write round-by-round
results to a CSV file.

Usage:
    python src/simulation/run_experiment.py --config configs/uniform.yaml --out results/uniform_run1.csv
"""

import argparse
import sys
from pathlib import Path

import pandas as pd
import yaml

sys.path.append(str(Path(__file__).resolve().parent.parent))

from simulation.agent import Agent
from simulation.mock_agent import MockAgent
from simulation.environment import ResourcePool
from simulation.enforcement import Enforcement
from simulation.graph import build_round_graph


def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def run(config: dict, mock: bool = False) -> pd.DataFrame:
    if mock:
        agents = [
            MockAgent(agent_id=a["id"], strategy=a.get("strategy", "random"))
            for a in config["agents"]
        ]
    else:
        agents = [
            Agent(agent_id=a["id"], model_name=a["model"], temperature=config.get("temperature", 0.7))
            for a in config["agents"]
        ]
    n_agents = len(agents)
    max_catch = config["max_catch_per_agent"]

    pool = ResourcePool(
        stock=config["initial_stock"],
        capacity=config["capacity"],
        regrowth_rate=config["regrowth_rate"],
        sustainability_threshold=config["sustainability_threshold"],
    )
    enforcement = Enforcement(
        sustainable_cap=config["sustainable_cap_per_agent"],
        fine_multiplier=config.get("fine_multiplier", 1.5),
        active=False,
    )

    round_graph = build_round_graph()

    round_num = 0
    for phase in config["phases"]:
        enforcement.set_active(phase["enforcement_active"])
        print(f"--- Phase: {phase['name']} (enforcement={enforcement.active}) ---")
        for _ in range(phase["n_rounds"]):
            round_num += 1
            state = {
                "agents": agents,
                "pool": pool,
                "enforcement": enforcement,
                "n_agents": n_agents,
                "max_catch": max_catch,
                "round_num": round_num,
                "phase": phase["name"],
                "requested": {},
                "actual": {},
                "fines": {},
            }
            round_graph.invoke(state)
            last = pool.history[-1]
            print(
                f"round {round_num:3d} | stock={last['stock']:6.1f} "
                f"| extracted={last['total_extraction']:6.1f} "
                f"| collapsed={last['collapsed']}"
            )
            if pool.is_collapsed():
                print(f"*** Resource collapsed at round {round_num} ({phase['name']}) ***")

    return pd.DataFrame(pool.history)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to YAML config")
    parser.add_argument("--out", required=True, help="Path to output CSV")
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Use heuristic MockAgents instead of real Ollama-backed LLMs (no GPU needed)",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    df = run(config, mock=args.mock)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"\nSaved {len(df)} rounds of results to {args.out}")


if __name__ == "__main__":
    main()
