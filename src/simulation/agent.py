"""An LLM-backed agent participating in the CPR game.

Each agent is a thin wrapper around a local Ollama model. It is given the current
public state (stock level, sustainable cap, whether enforcement is active, its own
past extractions and any sanctions it received) and asked to decide how much to
extract this round, as plain structured text that we parse defensively.
"""

import re
from dataclasses import dataclass, field

from langchain_core.messages import SystemMessage, HumanMessage

SYSTEM_PROMPT = """You are {agent_id}, one of {n_agents} fishers sharing a single fishing \
ground. Each round you decide how many fish to catch (an integer between 0 and {max_catch}). \
The fishing ground regenerates fish over time, but only if not overfished. If the group \
consistently overfishes, the ground collapses and everyone gets nothing in future rounds. \
{enforcement_clause}

Respond with ONLY a single integer: the number of fish you will catch this round. \
Do not explain your reasoning, just output the number."""

ENFORCEMENT_ON = (
    "There is currently a peer-enforcement system: agents who catch more than the "
    "agreed sustainable cap of {cap} fish may be fined by the group."
)
ENFORCEMENT_OFF = (
    "There is currently no enforcement system in place — no one will be fined for "
    "how much they catch."
)


@dataclass
class Agent:
    agent_id: str
    model_name: str
    temperature: float = 0.7
    _llm: object = field(init=False, repr=False, default=None)

    def __post_init__(self):
        # Imported lazily so that mock-mode runs (see mock_agent.py) don't require
        # langchain-ollama to be installed at all.
        from langchain_ollama import ChatOllama

        self._llm = ChatOllama(model=self.model_name, temperature=self.temperature)

    def decide_extraction(
        self,
        n_agents: int,
        max_catch: int,
        sustainable_cap: int,
        enforcement_active: bool,
        recent_context: str = "",
    ) -> float:
        clause = (
            ENFORCEMENT_ON.format(cap=sustainable_cap)
            if enforcement_active
            else ENFORCEMENT_OFF
        )
        system = SYSTEM_PROMPT.format(
            agent_id=self.agent_id,
            n_agents=n_agents,
            max_catch=max_catch,
            enforcement_clause=clause,
        )
        messages = [SystemMessage(content=system)]
        if recent_context:
            messages.append(HumanMessage(content=recent_context))
        else:
            messages.append(HumanMessage(content="How many fish do you catch this round?"))

        response = self._llm.invoke(messages)
        return self._parse_amount(response.content, max_catch)

    @staticmethod
    def _parse_amount(text: str, max_catch: int) -> float:
        """Defensively extract the first integer from the model's response."""
        match = re.search(r"-?\d+(\.\d+)?", text)
        if not match:
            return float(max_catch) / 2  # fallback: assume moderate extraction
        value = float(match.group())
        return max(0.0, min(float(max_catch), value))
