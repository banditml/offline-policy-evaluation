import json
import random
import time
import unittest
import uuid
from typing import Dict, List, Tuple

from faker import Faker
from faker.providers import internet

from models import (
    REWARD_TYPE_DELAYED,
    REWARD_TYPE_IMMEDIATE,
    Feedback,
    LegacyBase,
    Reward,
)


class TestFeedbackMappers(unittest.TestCase):
    def setUp(self):
        self.faker = Faker()
        self.faker.add_provider(internet)

    def test_from_decision(self):
        d = {
            "experiment_id": uuid.uuid4(),
            "mdp_id": uuid.uuid4(),
            "decision_id": uuid.uuid4(),
            "decision": random.choices(["a", "b", "c"]),
            "timestamp": time.time(),
            "variant_id": random.choices([1, 2, 3, 4]),
            "context": json.dumps({"ip_address": self.faker.ipv4()}),
            "score": random.random() * 10.0,
        }
        f = Feedback.from_decision("tesla", d)
        self.assertEqual("tesla", f.company)
        self.assert_match_bq_record(d, f)
        self.assertEqual("decision", f.event_category)
        self.assertEqual(d["timestamp"], f.timestamp)
        self.assertEqual(d["variant_id"], f.variant_id)
        self.assertEqual(d["context"], f.decision_context_json)
        self.assertEqual(d["score"], f.choice_score)

    def test_from_immediate_reward(self):
        metrics = {
            "price": random.random() * 100.0,
            "click": random.choice([0, 1]),
            "purchase": random.random() * 100.0,
        }
        r = self.make_reward(metrics)
        feedbacks = Feedback.from_reward("tesla", r)
        self.assertEqual(len(feedbacks), 3, "1 feedback per metric")
        for f in feedbacks:
            self.assertEqual("tesla", f.company)
            self.assertEqual(r["ts"], f.timestamp)
            self.assertEqual("reward", f.event_category)
            self.assertEqual(REWARD_TYPE_IMMEDIATE, f.reward_type)
        self.assert_metrics(metrics, feedbacks)

    def test_from_delayed_reward(self):
        products = ["prod-1", "prod-2", "prod-3"]
        metrics = {}
        for prod in products:
            metrics[prod] = {
                "price": random.random() * 100.0,
                "click": random.choice([0, 1]),
                "purchase": random.random() * 100.0,
            }
        r = self.make_reward(metrics, delayed=True)
        feedbacks = Feedback.from_reward("tesla", r)
        for f in feedbacks:
            self.assertEqual("tesla", f.company)
            self.assertEqual(r["ts"], f.timestamp)
            self.assertEqual(REWARD_TYPE_DELAYED, f.reward_type)
            self.assert_match_bq_record(r, f, delayed=True)

        for choice, metrics in metrics.items():
            match_feedbacks = [f for f in feedbacks if f.choice_id == choice]
            self.assertTrue(match_feedbacks)
            self.assert_metrics(metrics, match_feedbacks)

    def assert_metrics(self, metrics: Dict, feedbacks: List[Feedback]):
        for metric_name, metric_value in metrics.items():
            match_feedback = next(
                (f for f in feedbacks if f.event == metric_name), None
            )
            self.assertIsNotNone(match_feedback)
            self.assertEqual(metric_value, match_feedback.reward_value)

    def assert_match_bq_record(self, r: LegacyBase, f: Feedback, delayed:bool=False):
        self.assertEqual(r["experiment_id"], f.experiment_id)
        self.assertEqual(r["mdp_id"], f.mdp_id)
        if not delayed:
            self.assertEqual(r["decision_id"], f.decision_id)
            self.assertEqual(r["decision"], f.choice_id)

    @staticmethod
    def make_reward(metrics: Dict, delayed: bool = False) -> Reward:
        return {
            "experiment_id": uuid.uuid4(),
            "mdp_id": uuid.uuid4(),
            "decision_id": uuid.uuid4() if not delayed else None,
            "decision": random.choices(["a", "b", "c"]) if not delayed else None,
            "metrics": json.dumps(metrics),
            "ts": time.time(),
        }


def make_records(experiment_id, num_records) -> Tuple[LegacyBase, ...]:
    return tuple(LegacyBase(experiment_id=experiment_id) for _ in range(num_records))
