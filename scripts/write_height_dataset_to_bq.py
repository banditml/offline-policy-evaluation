"""
Script to fill bigquery table with intentionally simple dummy data.
Used to sanity test Bandit ML model implementations.

Usage:
    python scripts/write_height_dataset_to_bq.py
"""

import argparse
import json
import random
import time
import uuid

from google.cloud import bigquery
from google.oauth2 import service_account
import numpy as np

# Bigquery configs
PROJECT = "gradient-decision"
DATASET_ID = "gradient_app_staging"
DECISION_TABLE = "decisions"
REWARDS_TABLE = "rewards"
CREDS_PATH = "credentials/bq_creds.json"
IMMEDIATE_PARTITION = "$20200330"
DELAYED_PARTITION = "$20200401"

EXPERIMENT_ID = "test-experiment-height-prediction-v8"
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


def main(args):
    # initialize bigquery client
    credentials = service_account.Credentials.from_service_account_file(args.creds_path) if args.creds_path else None
    client = bigquery.Client(project=args.project, credentials=credentials)
    decision_table_immediate = client.get_table("{}.{}{}".format(args.dataset, args.decisions_table, IMMEDIATE_PARTITION))
    reward_table_immediate = client.get_table("{}.{}{}".format(args.dataset, args.rewards_table, IMMEDIATE_PARTITION))
    reward_table_delayed = client.get_table("{}.{}{}".format(args.dataset, args.rewards_table, DELAYED_PARTITION))

    # create decisions to insert into table
    decisions_to_insert, immediate_rewards_to_insert, end_of_mdp_rewards_to_insert = (
        [],
        [],
        [],
    )
    for i in range(NUM_GET_DECISION_CALLS):
        # pick random country, gender & year
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

        ######### decisions table schema #########
        # decision_id    <STRING>       REQUIRED
        # context        <JSON STRING>	REQUIRED
        # decision	     <INTEGER>      REQUIRED
        # experiment_id  <STRING>       NULLABLE
        # variation_id   <INTEGER>      NULLABLE
        # mdp_id         <STRING>       NULLABLE
        # ts             <INTEGER>      NULLABLE

        decisions_to_insert.append(
            {
                "decision_id": decision_id,
                "context": json.dumps(context),
                "decision": GENDER_MAP[gender],
                "experiment_id": EXPERIMENT_ID,
                "variation_id": random.randint(1, 2),
                "mdp_id": str(i),
                "ts": int(time.time()),
            }
        )

        ######### rewards table schema #########
        # decision_id    <STRING>       NULLABLE
        # decision	     <INTEGER>      NULLABLE
        # metrics        <JSON STRING>	NULLABLE
        # experiment_id  <STRING>       NULLABLE
        # mdp_id         <STRING>       NULLABLE
        # ts             <INTEGER>      NULLABLE

        # add some immediate rewards
        immediate_rewards_to_insert.append(
            {
                "decision_id": decision_id,
                "decision": GENDER_MAP[gender],
                "metrics": json.dumps({"height": height}),
                "experiment_id": EXPERIMENT_ID,
                "mdp_id": str(i),
                "ts": int(time.time()),
            }
        )

        # and add end of MDP rewards. in this problem all rewards are immediate
        # so add these as 0's to test the joining logic in training. in practice
        # end of MDP metrics' keys map to previous decisions
        end_of_mdp_rewards_to_insert.append(
            {
                "decision_id": None,
                "decision": None,
                "metrics": json.dumps({1: 0}),
                "experiment_id": EXPERIMENT_ID,
                "mdp_id": str(i),
                "ts": int(time.time()),
            }
        )

    # insert decisions into BQ table
    client.insert_rows(decision_table_immediate, decisions_to_insert)
    print(f"Successfully inserted {NUM_GET_DECISION_CALLS} decisions.")

    # insert immediate rewards into BQ table
    client.insert_rows(reward_table_immediate, immediate_rewards_to_insert)
    print(
        f"Successfully inserted {len(immediate_rewards_to_insert)} immediate rewards."
    )

    # insert end of MDP rewards into BQ table
    client.insert_rows(reward_table_delayed, end_of_mdp_rewards_to_insert)
    print(
        f"Successfully inserted {len(end_of_mdp_rewards_to_insert)} end of MDP rewards."
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--project",
        type=str,
        help="The ID of the GCP project where to create the Bigquery tables.",
        default=PROJECT,
    )
    parser.add_argument(
        "--dataset",
        type=str,
        help="Dataset id in which to create tables in BigQuery.",
        default=DATASET_ID,
    )
    parser.add_argument(
        "--decisions_table",
        type=str,
        help="Full name of the decision table (with the dataset).",
        default=DECISION_TABLE,
    )
    parser.add_argument(
        "--rewards_table",
        type=str,
        help="Full name of the rewards table (with the dataset).",
        default=REWARDS_TABLE,
    )
    parser.add_argument(
        "--creds_path",
        type=str,
        help="Path to a GCP credentials file",
        required=False,  # current account is ok
    )
    args = parser.parse_args()
    main(args)
