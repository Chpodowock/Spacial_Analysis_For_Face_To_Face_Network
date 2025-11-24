from entity import Entity
import numpy as np
import pandas as pd
from typing import Optional, Tuple


class Agent(Entity):
    def __init__(self, agent_id: str):
        super().__init__(entity_id=agent_id)
        self.trajectorie: Optional[pd.Series] = None  # pd.Series[datetime → area_id]
        self.entropy: Optional[float] = None
        self.max_entropy: Optional[float] = None
        self.normalized_entropy: Optional[float] = None

    def compute_entropy(self) -> None:

        if self.trajectorie is None or self.trajectorie.empty:
            self.entropy = self.max_entropy = self.normalized_entropy = 0.0
            return

        counts = self.trajectorie.value_counts(normalize=True)
        entropy = -(counts * np.log2(counts)).sum()
        max_entropy = np.log2(len(counts)) if len(counts) > 1 else 1
        normalized = entropy / max_entropy if max_entropy else 0

        self.entropy = entropy
        self.max_entropy = max_entropy
        self.normalized_entropy = normalized

    def __repr__(self) -> str:
        ent = f"{self.normalized_entropy:.2f}" if self.normalized_entropy is not None else "?"
        return f"<Agent {self.id} | NormEntropy={ent}>"
