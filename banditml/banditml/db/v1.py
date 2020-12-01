from typing_extensions import TypedDict

from .bq import Table

DECISIONS_TABLE = "decisions"
REWARDS_TABLE = "rewards"


class Base(TypedDict):
    experiment_id: str
    mdp_id: str
    decision_id: str
    decision: str
    ts: int


class Decision(Base):
    variation_id: int
    context: str
    score: float


class Reward(Base):
    metrics: str


class RewardTable(Table):
    name = REWARDS_TABLE
    partition_field = "_PARTITIONTIME"

    def mapper(self, bq_row) -> Reward:
        return {k: v for k, v in bq_row.items()}


class DecisionTable(Table):
    name = DECISIONS_TABLE
    partition_field = "_PARTITIONTIME"

    def mapper(self, bq_row) -> Decision:
        return {k: v for k, v in bq_row.items()}
