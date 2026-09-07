"""Peer-sanctioning mechanism: agents who extract above the sustainable cap are fined
when enforcement is active. Fines are tracked as a running penalty score per agent
(you can subtract this from a payoff metric during analysis).
"""

from dataclasses import dataclass, field


@dataclass
class Enforcement:
    sustainable_cap: float
    fine_multiplier: float = 1.5  # fine = multiplier * amount over the cap
    active: bool = False
    cumulative_fines: dict = field(default_factory=dict)

    def apply(self, extractions: dict) -> dict:
        """Return {agent_id: fine_amount} for this round. All zero if inactive."""
        fines = {agent_id: 0.0 for agent_id in extractions}
        if not self.active:
            return fines

        for agent_id, amount in extractions.items():
            if amount > self.sustainable_cap:
                over = amount - self.sustainable_cap
                fine = over * self.fine_multiplier
                fines[agent_id] = fine
                self.cumulative_fines[agent_id] = (
                    self.cumulative_fines.get(agent_id, 0.0) + fine
                )
        return fines

    def set_active(self, active: bool) -> None:
        self.active = active
