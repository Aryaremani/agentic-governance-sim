"""Common-pool-resource (CPR) environment: a shared, regenerating resource pool.

The "fishery" model: each round, the stock regenerates by a logistic growth rule,
then agents extract from it. If total extraction exceeds the stock, extraction is
scaled down proportionally (agents can't take more than exists).
"""

from dataclasses import dataclass, field


@dataclass
class ResourcePool:
    stock: float
    capacity: float = 100.0
    regrowth_rate: float = 0.20  # logistic growth rate per round
    sustainability_threshold: float = 20.0  # below this = "collapsed"
    refuge_fraction: float = 0.05  # fraction of capacity that is never harvestable
    history: list = field(default_factory=list)

    @property
    def refuge(self) -> float:
        """A small unharvestable population (breeding refuge). Without this, logistic
        regrowth is exactly zero once stock hits 0, making collapse irreversible even
        if extraction pressure later drops — which is unrealistic and makes for a
        boring/uninformative simulation (everything just flatlines at zero forever)."""
        return self.capacity * self.refuge_fraction

    def regenerate(self) -> None:
        """Logistic regrowth: stock grows toward capacity, slower near the cap."""
        growth = self.regrowth_rate * self.stock * (1 - self.stock / self.capacity)
        self.stock = max(0.0, min(self.capacity, self.stock + growth))

    def extract(self, requested: dict) -> dict:
        """Apply extraction requests. Extraction can never pull stock below the refuge
        floor, and is scaled down proportionally across agents if total demand exceeds
        what's harvestable.

        Args:
            requested: {agent_id: amount_requested}

        Returns:
            {agent_id: amount_actually_extracted}
        """
        total_requested = sum(requested.values())
        harvestable = max(0.0, self.stock - self.refuge)

        if total_requested <= 0 or harvestable <= 0:
            return {agent_id: 0.0 for agent_id in requested}

        scale = min(1.0, harvestable / total_requested)
        actual = {agent_id: amt * scale for agent_id, amt in requested.items()}
        self.stock -= sum(actual.values())
        self.stock = max(self.refuge, self.stock)
        return actual

    def is_collapsed(self) -> bool:
        return self.stock < self.sustainability_threshold

    def log_round(self, round_num: int, phase: str, extractions: dict, sanctions: dict) -> None:
        self.history.append(
            {
                "round": round_num,
                "phase": phase,
                "stock": self.stock,
                "total_extraction": sum(extractions.values()),
                "collapsed": self.is_collapsed(),
                **{f"extract_{k}": v for k, v in extractions.items()},
                **{f"sanction_{k}": v for k, v in sanctions.items()},
            }
        )
