import io
import itertools
import json
import logging
import numbers
import os
from collections import defaultdict
from datetime import datetime, timedelta
from typing import Any, Iterable, List, Dict, NamedTuple, Sequence, Optional

from google.cloud import bigquery
from google.oauth2 import service_account

# after move to Python 3.8 this should be imported from `typing`.
from typing_extensions import TypedDict


logger = logging.getLogger(__name__)

REWARD_TYPE_IMMEDIATE = "immediate"
REWARD_TYPE_DELAYED = "delayed"


class LegacyBase(TypedDict):
    experiment_id: str
    mdp_id: str
    decision_id: str
    decision: str


class Decision(LegacyBase):
    timestamp: int
    variant_id: int
    context: str
    score: float


class Reward(LegacyBase):
    metrics: str
    ts: int


class Feedback(NamedTuple):
    timestamp: int
    company: str
    experiment_id: str
    mdp_id: str
    event: str
    event_category: str
    choice_id: Optional[str]
    choice_score: Optional[float]
    variant_id: Optional[int]
    variant_slug: Optional[str]
    decision_id: Optional[str]
    decision_context_json: Optional[str]
    reward_type: Optional[str]
    reward_value: Optional[float]

    def to_bq_record(self) -> LegacyBase:
        return dict(self)

    @classmethod
    def from_decision(cls, company: str, decision: Decision) -> "Feedback":
        """
        Maps a legacy `decision` record to a new `feedback` record.
        `variant_slug` is unmappable without API access (OSS) or DB access (internal).
        """
        return Feedback(
            timestamp=decision["timestamp"],
            company=company,
            experiment_id=decision["experiment_id"],
            mdp_id=decision["mdp_id"],
            event="decision",
            event_category="decision",
            choice_id=decision["decision"],
            choice_score=decision["score"],
            variant_id=decision["variant_id"],
            variant_slug=None,
            decision_id=decision["decision_id"],
            decision_context_json=decision["context"],
            reward_type=None,
            reward_value=None,
        )

    @classmethod
    def from_reward(cls, company: str, reward: Reward) -> List["Feedback"]:
        """
        Maps a legacy `reward` record to multiple `feedback` records.
        """

        def create_feedback(reward_key, reward_value) -> "Feedback":
            # uses `choice_id` and `reward_type` from scope
            return Feedback(
                timestamp=reward["ts"],
                company=company,
                experiment_id=reward["experiment_id"],
                mdp_id=reward["mdp_id"],
                event=reward_key,
                event_category="reward",
                choice_id=choice_id,
                choice_score=None,
                variant_id=None,
                variant_slug=None,
                decision_id=None,
                decision_context_json=None,
                reward_type=reward_type,
                reward_value=reward_value,
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
                    feedbacks.append(
                        create_feedback(inner_metric_key, inner_metric_value)
                    )
            else:
                assert isinstance(metric_key, str)
                assert isinstance(metric_value, numbers.Number)
                choice_id = reward.get("decision")
                feedbacks.append(create_feedback(metric_key, metric_value))
        return feedbacks
