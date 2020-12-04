"""
Script to create BigQuery tables following training data schema.

Usage:
    python -m scripts/create_bq_tables.py
"""

import argparse
import json
from typing import Dict, List

import google.api_core.exceptions as gexceptions
from google.cloud import bigquery
from google.oauth2 import service_account

# temp code
PROJECT = "gradient-decision"
DATASET_ID = "bandit_app"


def create_dataset(
    client: bigquery.Client, dataset_id: str, description: str, location: str
) -> None:
    """Creates a dataset in GCP.

    Args:
        client: The client used to create the dataset. Client should have a project defined.
        dataset_id: The dataset id of the dataset to create.
        description: The description of the dataset to create.
        location: The GCP location of the dataset to create.

    """
    dataset = bigquery.Dataset("{}.{}".format(client.project, dataset_id))
    dataset.description = description
    dataset.location = location
    try:
        client.create_dataset(dataset)
        print("Created dataset {}.{}".format(client.project, dataset_id))
    except gexceptions.Conflict:
        print(
            "Dataset {} already existing in project {}. Skipping dataset creation...".format(
                dataset_id, client.project
            )
        )


def create_table(
    client: bigquery.Client, dataset_id: str, table_id: str, fields: List[Dict]
):
    dataset = bigquery.Dataset("{}.{}".format(client.project, dataset_id))
    table_ref = dataset.table(table_id)
    table = bigquery.Table(table_ref, schema=fields)
    table.time_partitioning = bigquery.table.TimePartitioning()  # partition by day

    try:
        table = client.create_table(table)
        print(
            "Created table {}.{}.{}".format(
                table.project, table.dataset_id, table.table_id
            )
        )
    except gexceptions.GoogleAPIError as e:
        print("Table {} could not be created: {}. Skipping...".format(table_id, e))


def main(args) -> None:
    """Create BigQuery tables"""
    if args.creds_path:
        credentials = service_account.Credentials.from_service_account_file(
            args.creds_path
        )
    else:
        credentials = None
    client = bigquery.Client(project=args.project, credentials=credentials)
    create_dataset(
        client, args.dataset, args.dataset_description, args.dataset_location
    )

    # create decision table
    with open("scripts/schemas/training_data.json") as data_schema:
        fields = json.load(data_schema)
        create_table(client, args.dataset, "training_data", fields)

    print("Training dataset table created.")


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
        "--creds_path",
        type=str,
        help="Path to a GCP credentials file",
        required=False,  # current account is ok
    )
    parser.add_argument(
        "--dataset_location",
        type=str,
        help="GCP location of the dataset.",
        default="US",
    )
    parser.add_argument(
        "--dataset_description",
        type=str,
        help="Description of the dataset. Only used if the dataset does not already exist.",
        default="Bandit ML dataset.",
    )

    args = parser.parse_args()
    main(args)
