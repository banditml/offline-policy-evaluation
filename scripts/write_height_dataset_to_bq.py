"""
Script to fill bigquery table with intentionally simple dummy data.
Used to sanity test Bandit ML model implemenations.

Usage:
    python scripts/write_height_dataset_to_bq.py
"""

from collections import defaultdict
import json
import random
import sys
import uuid

from google.cloud import bigquery
from google.oauth2 import service_account
import numpy as np


# Bigquery configs
PROJECT = "gradient-decision"
DECISION_TABLE_NAME = "gradient_app_staging.decisions"
REWARDS_TABLE_NAME = "gradient_app_staging.rewards"
CREDS_PATH = "credentials/bq_creds.json"

EXPERIMENT_ID = "test-experiment-height-prediction-v2"
CURRENT_HEIGHT_DISTRIBUTIONS = {
    "usa": {
        "id": 1,
        "male": {"mean": 175.3, "stddev": 5},
        "female": {"mean": 161.5, "stddev": 4},
    },
    "china": {
        "id": 2,
        "male": {"mean": 169.5, "stddev": 4},
        "female": {"mean": 158, "stddev": 3},
    },
    "india": {
        "id": 3,
        "male": {"mean": 166.3, "stddev": 3},
        "female": {"mean": 158.5, "stddev": 2},
    },
    "serbia": {
        "id": 4,
        "male": {"mean": 182, "stddev": 5},
        "female": {"mean": 166.8, "stddev": 4},
    },
    "norway": {
        "id": 5,
        "male": {"mean": 179.7, "stddev": 5},
        "female": {"mean": 167.1, "stddev": 4},
    },
}
GENDER_MAP = {"male": 1, "female": 2}
CURRENT_YEAR = 2020
YEARLY_MEAN_CM_ADJUSTMENTS = 0.25
POSSIBLE_YEARS = range(CURRENT_YEAR - 50, CURRENT_YEAR + 1)

NUM_GET_DECISION_CALLS = 10000


def main():
    # intialize bigquery client
    credentials = service_account.Credentials.from_service_account_file(CREDS_PATH)
    client = bigquery.Client(project=PROJECT, credentials=credentials)
    decision_table = client.get_table(DECISION_TABLE_NAME)
    reward_table = client.get_table(REWARDS_TABLE_NAME)

    # create decisions to insert into table
    decisions_to_insert, rewards_to_insert = [], []
    for i in range(NUM_GET_DECISION_CALLS):

        # pick random country, genderm & year
        country = random.choice(list(CURRENT_HEIGHT_DISTRIBUTIONS.keys()))
        gender = random.choice(["male", "female"])
        year = random.choice(POSSIBLE_YEARS)

        mean_height_adjustment = (CURRENT_YEAR - year) * YEARLY_MEAN_CM_ADJUSTMENTS
        mu = (
            CURRENT_HEIGHT_DISTRIBUTIONS[country][gender]["mean"]
            - mean_height_adjustment
        )
        sigma = CURRENT_HEIGHT_DISTRIBUTIONS[country][gender]["stddev"]
        height = np.random.normal(mu, sigma, 1)[0]

        context = {
            "country": CURRENT_HEIGHT_DISTRIBUTIONS[country][
                "id"
            ],  # idlist feature or categorical
            "year": year,  # numeric feature
        }

        decision_id = str(uuid.uuid4())

        decisions_to_insert.append(
            {
                "decision_id": decision_id,
                "context": json.dumps(context),
                "decision": GENDER_MAP[gender],
                "experiment_id": EXPERIMENT_ID,
            }
        )

        rewards_to_insert.append(
            {"decision_id": decision_id, "metrics": json.dumps({"height": height})}
        )

    # insert decisions into BQ table
    client.insert_rows(decision_table, decisions_to_insert)
    print(f"Successfully inserted {NUM_GET_DECISION_CALLS} decisions.")

    # insert rewards into BQ table
    client.insert_rows(reward_table, rewards_to_insert)
    print(f"Successfully inserted {len(rewards_to_insert)} rewards.")


if __name__ == "__main__":
    main()
