"""
Script to backfill bigquery table with rewards.

Usage:
    python scripts/backfill_rewards.py \
        --csv_path path/to/events.csv \
        --company_name dunder_mifflin \
        --experiment_id backfill

Assumes CSV schema of:
    | ts <int> | session_id <str> | item_id <str> | event <str> |
"""

import argparse
import itertools
from typing import Any, Iterable

import pandas as pd
from google.cloud import bigquery
from google.oauth2 import service_account

CHUNK_WRITE_SIZE = 10000
EVENT_MAPPING = {}


def chunker(iterable: Iterable[Any], size: int) -> [(None, None), (int, Iterable[Any])]:
    # https://stackoverflow.com/a/8998040
    it = iter(iterable)
    i = 0
    while True:
        # tuple consumes from the islice up to `size`
        chunk = tuple(itertools.islice(it, size))
        if not chunk:
            return
        yield i, chunk
        i += 1


def main(args):
    # initialize bigquery client
    if args.creds_path:
        credentials = service_account.Credentials.from_service_account_file(
            args.creds_path
        )
    else:
        credentials = None

    client = bigquery.Client(project=args.project, credentials=credentials)
    table = client.get_table("{}.{}".format(args.dataset, args.table_name))

    df = pd.read_csv(args.csv_path)
    # replace NaNs with Nones
    df = df.where(pd.notnull(df), None)

    events_to_insert = []
    for _, row in df.iterrows():
        events_to_insert.append(
            {
                "timestamp": row["ts"],
                "company": args.company_name,
                "experiment_id": args.experiment_id,
                "type": "reward",
                "reward_metric": row["event"],
                "mdp_id": row["session_id"],
                "choice_id": row["item_id"],
            }
        )

    all_errors = []
    for i, chunk in chunker(events_to_insert, CHUNK_WRITE_SIZE):
        print(f"Inserting chunk {i}...")
        errors = client.insert_rows(
            table=table, rows=chunk, skip_invalid_rows=True, ignore_unknown_values=False
        )
        for error in errors:
            # unchunk the error index for caller parsing.
            error["index"] += i * CHUNK_WRITE_SIZE
        all_errors.extend(errors)

    num_fails = len(all_errors)
    num_successes = len(events_to_insert) - num_fails

    print(f"Successfully inserted {num_successes} events.")

    if num_fails > 0:
        print(f"Failed to insert {num_fails} events.")
        print("Preview of errors:")
        print(f"{all_errors[:5]}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--csv_path", type=str, help="Path to rewards csv.", required=True
    )
    parser.add_argument(
        "--company_name", type=str, help="Company identifier.", required=True
    )
    parser.add_argument(
        "--experiment_id",
        type=str,
        help="Associated experiment for logs.",
        default=None,
    )
    parser.add_argument(
        "--project",
        type=str,
        help="The ID of the GCP project where to create the Bigquery tables.",
        default="gradient-decision",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        help="Dataset id in which to create tables in BigQuery.",
        default="bandit_app",
    )
    parser.add_argument(
        "--table_name",
        type=str,
        help="Name of the table to insert into.",
        default="feedback",
    )
    parser.add_argument(
        "--creds_path",
        type=str,
        help="Path to a GCP credentials file",
        default="credentials/bq_creds.json",
    )
    args = parser.parse_args()
    main(args)
