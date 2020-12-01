from __future__ import unicode_literals

import json
import numbers
import time
from typing import Dict, List, Optional

from google.cloud import bigquery

from .bq import Table
from .v1 import Decision, Reward

FEEDBACK_TABLE = "feedback"
REWARD_TYPE_IMMEDIATE = "immediate"
REWARD_TYPE_DELAYED = "delayed"


SCHEMA = [
    bigquery.SchemaField("timestamp", "TIMESTAMP", mode="REQUIRED"),
    bigquery.SchemaField("company", "STRING", mode="NULLABLE"),
    bigquery.SchemaField("experiment_id", "STRING", mode="NULLABLE"),
    bigquery.SchemaField("mdp_id", "STRING", mode="NULLABLE"),
    bigquery.SchemaField("type", "STRING", mode="NULLABLE"),
    bigquery.SchemaField("choice_id", "STRING", mode="NULLABLE"),
    bigquery.SchemaField("choice_score", "FLOAT64", mode="NULLABLE"),
    bigquery.SchemaField("variant_id", "INT64", mode="NULLABLE"),
    bigquery.SchemaField("variant_slug", "STRING", mode="NULLABLE"),
    bigquery.SchemaField("decision_id", "STRING", mode="NULLABLE"),
    bigquery.SchemaField("decision_context_json", "STRING", mode="NULLABLE"),
    bigquery.SchemaField("reward_type", "STRING", mode="NULLABLE"),
    bigquery.SchemaField("reward_metric", "STRING", mode="NULLABLE"),
    bigquery.SchemaField("reward_value", "FLOAT64", mode="NULLABLE"),
]


class Feedback:
    def __init__(self, **kwargs):
        self.timestamp: int = kwargs.get("timestamp", time.time())
        self.company: str = kwargs.get("company", None)
        self.experiment_id: str = kwargs.get("experiment_id", None)
        self.mdp_id: str = kwargs.get("mdp_id", None)
        self.type: str = kwargs.get("type", None)
        self.choice_id: Optional[str] = kwargs.get("choice_id", None)
        self.choice_score: Optional[float] = kwargs.get("choice_score", None)
        self.variant_id: Optional[int] = kwargs.get("variant_id", None)
        self.variant_slug: Optional[str] = kwargs.get("variant_slug", None)
        self.decision_id: Optional[str] = kwargs.get("decision_id", None)
        self.decision_context_json: Optional[str] = kwargs.get(
            "decision_context_json", None
        )
        self.reward_type: Optional[str] = kwargs.get("reward_type", None)
        self.reward_metric: Optional[str] = kwargs.get("reward_metric", None)
        self.reward_value: Optional[float] = kwargs.get("reward_value", None)

    def to_dict(self) -> Dict:
        return self.__dict__

    @classmethod
    def from_decision(cls, company: str, decision: Decision) -> "Feedback":
        """
        Maps a v1 `decision` record to a v2 `feedback` record.
        `variant_slug` is unmappable without API access (OSS) or DB access (internal).
        """
        return Feedback(
            timestamp=decision.get("ts", None),
            company=company,
            type="decision",
            choice_id=decision.get("decision", None),
            choice_score=decision.get("score", None),
            decision_context_json=decision.get("context", None),
            variant_id=decision.get("variation_id", None),
            **decision,
        )

    @classmethod
    def from_reward(cls, company: str, reward: Reward) -> List["Feedback"]:
        """
        Maps a legacy `reward` record to multiple `feedback` records.
        """

        def create_feedback(reward_key, reward_value) -> "Feedback":
            # uses `choice_id` and `reward_type` from scope
            return Feedback(
                timestamp=reward.get("ts", None),
                company=company,
                type="reward",
                choice_id=choice_id,
                reward_type=reward_type,
                reward_metric=reward_key,
                reward_value=reward_value,
                **reward,
            )

        feedbacks = []
        reward_type = REWARD_TYPE_IMMEDIATE
        if reward.get("decision") is None and reward.get("decision_id") is None:
            reward_type = REWARD_TYPE_DELAYED
        metrics = json.loads(reward["metrics"])
        for metric_key, metric_value in metrics.items():
            if reward_type == REWARD_TYPE_DELAYED:
                assert isinstance(metric_key, str)
                assert isinstance(metric_value, dict)
                choice_id = metric_key
                inner_metrics = metric_value
                for inner_metric_key, inner_metric_value in inner_metrics.items():
                    assert isinstance(inner_metric_key, str)
                    if not isinstance(inner_metric_value, numbers.Number):
                        continue
                    feedbacks.append(
                        create_feedback(inner_metric_key, inner_metric_value)
                    )
            else:
                assert isinstance(metric_key, str)
                if not isinstance(metric_value, numbers.Number):
                    continue
                choice_id = reward.get("decision")
                feedbacks.append(create_feedback(metric_key, metric_value))
        return feedbacks


class FeedbackTable(Table):
    name = FEEDBACK_TABLE
    model = Feedback

    @classmethod
    def create(
        cls, client: bigquery.Client, project_id: str, dataset_id: str, **kwargs
    ) -> "FeedbackTable":
        return super().create(
            client, project_id, dataset_id, schema=SCHEMA, partition=True
        )

    def mapper(self, bq_row) -> Feedback:
        return Feedback(**{k: v for k, v in bq_row.items()})
