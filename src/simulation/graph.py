"""LangGraph wiring for a single round of the CPR game.

Graph shape for one round:

    decide_extractions -> settle_round -> regenerate_pool -> END

The graph is invoked once per round from run_experiment.py, which loops over rounds
and phases and swaps `enforcement.active` between phases.
"""

from typing import TypedDict

from langgraph.graph import StateGraph, END

from .agent import Agent
from .environment import ResourcePool
from .enforcement import Enforcement


class RoundState(TypedDict):
    agents: list          # list[Agent], not mutated
    pool: ResourcePool
    enforcement: Enforcement
    n_agents: int
    max_catch: int
    round_num: int
    phase: str
    requested: dict
    actual: dict
    fines: dict


def decide_extractions(state: RoundState) -> RoundState:
    requested = {}
    for agent in state["agents"]:
        amount = agent.decide_extraction(
            n_agents=state["n_agents"],
            max_catch=state["max_catch"],
            sustainable_cap=state["enforcement"].sustainable_cap,
            enforcement_active=state["enforcement"].active,
        )
        requested[agent.agent_id] = amount
    state["requested"] = requested
    return state


def settle_round(state: RoundState) -> RoundState:
    pool: ResourcePool = state["pool"]
    enforcement: Enforcement = state["enforcement"]

    actual = pool.extract(state["requested"])
    fines = enforcement.apply(actual)

    pool.log_round(state["round_num"], state["phase"], actual, fines)

    state["actual"] = actual
    state["fines"] = fines
    return state


def regenerate_pool(state: RoundState) -> RoundState:
    state["pool"].regenerate()
    return state


def build_round_graph():
    graph = StateGraph(RoundState)
    graph.add_node("decide_extractions", decide_extractions)
    graph.add_node("settle_round", settle_round)
    graph.add_node("regenerate_pool", regenerate_pool)

    graph.set_entry_point("decide_extractions")
    graph.add_edge("decide_extractions", "settle_round")
    graph.add_edge("settle_round", "regenerate_pool")
    graph.add_edge("regenerate_pool", END)

    return graph.compile()
