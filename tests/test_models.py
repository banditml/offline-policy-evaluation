import json
import random
import time
import unittest
import uuid
from typing import Dict, List

from faker import Faker
from faker.providers import internet

from banditml.banditml.db.v1 import Base as LegacyBase
from banditml.banditml.db.v2 import (
    REWARD_TYPE_DELAYED,
    REWARD_TYPE_IMMEDIATE,
    Feedback,
    Reward,
)


class TestFeedbackMappers(unittest.TestCase):
    def setUp(self):
        self.faker = Faker()
        self.faker.add_provider(internet)

    def test_to_from_decision(self):
        expected = Feedback(
            timestamp=int(time.time()),
            company="acme",
            experiment_id="anvils",
            mdp_id="wil-e-coyote",
            type="decision",
            choice_id="defective-anvil",
            choice_score=-1,
            variant_id=1,
            variant_slug="episode-1",
            decision_id="all-anvils",
            decision_context_json="{}",
            reward_type=None,
            reward_metric=None,
            reward_value=None,
        )
        self.assertEqual(expected.to_dict(), Feedback(**expected.to_dict()).to_dict())

    def test_from_decision(self):
        d = {
            "experiment_id": uuid.uuid4(),
            "mdp_id": uuid.uuid4(),
            "decision_id": uuid.uuid4(),
            "decision": random.choices(["a", "b", "c"]),
            "ts": time.time(),
            "variation_id": random.choices([1, 2, 3, 4]),
            "context": json.dumps({"ip_address": self.faker.ipv4()}),
            "score": random.random() * 10.0,
        }
        f = Feedback.from_decision("tesla", d)
        self.assertEqual("tesla", f.company)
        self.assertEqual("decision", f.type)
        self.assert_match_bq_record(d, f)
        self.assertEqual(d["variation_id"], f.variant_id)
        self.assertEqual(d["context"], f.decision_context_json)
        self.assertEqual(d["score"], f.choice_score)

    def test_from_immediate_reward(self):
        metrics = {
            "price": random.random() * 100.0,
            "click": random.choice([0, 1]),
            "purchase": random.random() * 100.0,
            "bad_metric": "haha strings are da best",
        }
        r = self.make_reward(metrics)
        feedbacks = Feedback.from_reward("tesla", r)
        self.assertEqual(len(feedbacks), 3, "1 feedback per metric")
        for f in feedbacks:
            self.assertEqual("tesla", f.company)
            self.assertEqual("reward", f.type)
            self.assertEqual(REWARD_TYPE_IMMEDIATE, f.reward_type)
        metrics.pop("bad_metric")
        self.assert_metrics(metrics, feedbacks)

    def test_from_delayed_reward(self):
        products = ["prod-1", "prod-2", "prod-3"]
        metrics = {}
        for prod in products:
            metrics[prod] = {
                "price": random.random() * 100.0,
                "click": random.choice([0, 1]),
                "purchase": random.random() * 100.0,
                "bad_metric": "haha strings are way better than non-strings",
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
            metrics.pop("bad_metric")
            self.assert_metrics(metrics, match_feedbacks)

    def assert_metrics(self, metrics: Dict, feedbacks: List[Feedback]):
        for metric_name, metric_value in metrics.items():
            match_feedback = next(
                (f for f in feedbacks if f.reward_metric == metric_name), None
            )
            self.assertIsNotNone(match_feedback, f"missing metric for {metric_name}")
            self.assertEqual(metric_value, match_feedback.reward_value)

    def assert_match_bq_record(self, r: LegacyBase, f: Feedback, delayed: bool = False):
        self.assertEqual(r["experiment_id"], f.experiment_id)
        self.assertEqual(r["mdp_id"], f.mdp_id)
        self.assertEqual(r["ts"], f.timestamp)
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
