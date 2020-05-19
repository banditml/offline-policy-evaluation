"""
Script used to check BigQuery tables before training for common errors.

Usage:
    python scripts/check_bq_data.py \
        --project my-bq-project \
        --dataset my-bq-app \
        --creds_path credentials/bq_creds.json \
        --start_ds 2020-04-22 \
        --experiment_id my-experiment-id
"""

import argparse

from google.cloud import bigquery
from google.oauth2 import service_account


def query_to_df(client, query: str):
    return client.query(query).result().to_dataframe()


def main(args):
    credentials = service_account.Credentials.from_service_account_file(args.creds_path)
    client = bigquery.Client(project=args.project, credentials=credentials)

    decisions_table = f"{args.project}.{args.dataset}.decisions"
    rewards_table = f"{args.project}.{args.dataset}.rewards"

    print("Checking decisions table...")
    print("-" * 30)
    print("\n")

    print("Checking schema of decisions table\n")
    df = query_to_df(
        client,
        f"""
        select
            *
        from `{args.project}.{args.dataset}`.INFORMATION_SCHEMA.COLUMNS
        where table_name = "{decisions_table}"
        """,
    )
    print(df)
    print("\n")

    print("Checking number of samples in decisions table\n")
    df = query_to_df(
        client,
        f"""
        select
            count(*) num_samples
        from `{decisions_table}`
        where date(_partitiontime) >= "{args.start_ds}"
            and experiment_id = "{args.experiment_id}"
        """,
    )
    print(df)
    print("\n")

    print("Checking number of unique decisions in table\n")
    df = query_to_df(
        client,
        f"""
        select
            count(distinct decision) num_decisions
        from `{decisions_table}`
        where date(_partitiontime) >= "{args.start_ds}"
            and experiment_id = "{args.experiment_id}"
        """,
    )
    print(df)
    print("\n")

    print("Checking how many decision rows per decision id there are\n")
    df = query_to_df(
        client,
        f"""
        select
            decision_id,
            count(*) num_decisions
        from `{decisions_table}`
        where date(_partitiontime) >= "{args.start_ds}"
            and experiment_id = "{args.experiment_id}"
        group by 1
        order by 2 desc
        limit 10
        """,
    )
    print(df)
    print("\n")

    print("Checking that rows are unique on concat(decision_id, decision)\n")
    df = query_to_df(
        client,
        f"""
        select
            count(distinct concat(decision_id, decision)) unique_pairs,
            count(concat(decision_id, decision)) all_pairs
        from `{decisions_table}`
        where date(_partitiontime) >= "{args.start_ds}"
            and experiment_id = "{args.experiment_id}"
        """,
    )
    print(df)
    print("\n")

    print(
        "Check how many decisions are made on average per mdp - this is "
        "how `sequential` the problem is\n"
    )
    df = query_to_df(
        client,
        f"""
        select
            mdp_id,
            count(*) decisions_per_mdp
        from `{decisions_table}`
        where date(_partitiontime) >= "{args.start_ds}"
            and experiment_id = "{args.experiment_id}"
        group by 1
        order by 2 desc
        limit 10
        """,
    )
    print(df)
    print("\n")

    print(
        "Checking if position feature appears in context. This helps de-bias "
        "learning for ranking problems\n"
    )
    df = query_to_df(
        client,
        f"""
        select
            JSON_EXTRACT(context, '$.position') position
        from `{decisions_table}`
        where date(_partitiontime) >= "{args.start_ds}"
            and experiment_id = "{args.experiment_id}"
        limit 10
        """,
    )
    print(df)
    print("\n")

    print("Checking the number of features per context and the frequencies\n")
    udf = '"""return Object.keys(JSON.parse(input)).length"""'
    df = query_to_df(
        client,
        f"""
        create temp function numJsonKeys(input string)
        returns int64
        language js AS {udf};

        select
            numJsonKeys(context) num_features,
            count(*) count
        from `{decisions_table}`
        where date(_partitiontime) >= "{args.start_ds}"
            and experiment_id = "{args.experiment_id}"
        group by 1
        order by 2 desc
        limit 10
        """,
    )
    print(df)
    print("\n")

    print("Check a few example contexts with visual inspection\n")
    df = query_to_df(
        client,
        f"""
        select
            context
        from `{decisions_table}`
        where date(_partitiontime) >= "{args.start_ds}"
            and experiment_id = "{args.experiment_id}"
        limit 10
        """,
    )
    print(df)
    print("\n")

    print("\n")
    print("Checking rewards table...")
    print("-" * 30)
    print("\n")

    print("Checking schema of rewards table\n")
    df = query_to_df(
        client,
        f"""
        select
            *
        from `{args.project}.{args.dataset}`.INFORMATION_SCHEMA.COLUMNS
        where table_name = "{rewards_table}"
        """,
    )
    print(df)
    print("\n")

    print("Checking number of samples in rewards table\n")
    df = query_to_df(
        client,
        f"""
        select
            count(*) num_samples
        from `{rewards_table}`
        where date(_partitiontime) >= "{args.start_ds}"
        and experiment_id = "{args.experiment_id}"
        """,
    )
    print(df)
    print("\n")

    print("Check how many immediate reward rows there are\n")
    df = query_to_df(
        client,
        f"""
        select
            count(*) num_immediate_rewards
        from `{rewards_table}`
        where date(_partitiontime) >= "{args.start_ds}"
            and experiment_id = "{args.experiment_id}"
            and decision_id is not null
            and decision is not null
        """,
    )
    print(df)
    print("\n")

    print(
        "Check how many rewards have no decision_id, decision, and mpd_id - "
        "should be 0%\n"
    )
    df = query_to_df(
        client,
        f"""
        select
            count(*) busted_reward_rows
        from `{rewards_table}`
        where date(_partitiontime) >= "{args.start_ds}"
            and experiment_id = "{args.experiment_id}"
            and decision_id is null
            and decision is null
            and mdp_id is null
        """,
    )
    print(df)
    print("\n")

    print("Visually inspect a few of the immediate rewards\n")
    df = query_to_df(
        client,
        f"""
        select
            *
        from `{rewards_table}`
        where date(_partitiontime) >= "{args.start_ds}"
            and experiment_id = "{args.experiment_id}"
            and decision_id is not null
            and decision is not null
        limit 10
        """,
    )
    print(df)
    print("\n")

    print("Output the set of keys present in immediate rewards\n")
    udf = '"""return Object.keys(JSON.parse(input))"""'
    df = query_to_df(
        client,
        f"""
        create temp function getJsonKeys(input string)
        returns array<string>
        language js AS {udf};

        with immediate_reward_keys AS (
          select
            getJsonKeys(metrics) keys
          from `{rewards_table}`
          where decision_id is not null
            and decision is not null
            and date(_partitiontime) >= "{args.start_ds}"
            and experiment_id = "{args.experiment_id}"
        )

        select
          distinct reward_key
        from immediate_reward_keys, unnest(keys) as reward_key
        """,
    )
    print(df)
    print("\n")

    print(
        "Check what percent of immediate rewards join to decisions - should be 100%\n"
    )
    df = query_to_df(
        client,
        f"""
        with r as (
            select
                decision_id,
                decision
            from `{rewards_table}`
            where date(_partitiontime) >= "{args.start_ds}"
                and experiment_id = "{args.experiment_id}"
                and decision_id is not null
                and decision is not null
        ), d as (
            select
                decision_id,
                decision
            from `{decisions_table}`
            where date(_partitiontime) >= "{args.start_ds}"
                and experiment_id = "{args.experiment_id}"
        )
        select
            sum(case when d.decision is not null then 1 else 0 end) /
            count(*) percent_imm_rewards_with_associated_decision
        from r
        left join d on d.decision_id = r.decision_id
            and d.decision = r.decision
        """,
    )
    print(df)
    print("\n")

    print("Check number of immediate rewards per decision id\n")
    df = query_to_df(
        client,
        f"""
        with r as (
            select
                decision_id,
                decision
            from `{rewards_table}`
            where date(_partitiontime) >= "{args.start_ds}"
                and experiment_id = "{args.experiment_id}"
                and decision_id is not null
                and decision is not null
        ), d as (
            select
                decision_id,
                decision
            from `{decisions_table}`
            where date(_partitiontime) >= "{args.start_ds}"
                and experiment_id = "{args.experiment_id}"
        )
        select
            d.decision_id,
            sum(case when r.decision is not null then 1 else 0 end) num_immediate_rewards
        from d
        left join r on d.decision_id = r.decision_id
            and d.decision = r.decision
        group by 1
        order by 2 desc
        limit 10
        """,
    )
    print(df)
    print("\n")

    print("Check how many delayed reward rows there are\n")
    df = query_to_df(
        client,
        f"""
        select
            count(*) num_delayed_rewards
        from `{rewards_table}`
        where date(_partitiontime) >= "{args.start_ds}"
            and experiment_id = "{args.experiment_id}"
            and decision_id is null
            and decision is null
        """,
    )
    print(df)
    print("\n")

    print("Visually inspect a few of the delayed rewards\n")
    df = query_to_df(
        client,
        f"""
        select
            *
        from `{rewards_table}`
        where date(_partitiontime) >= "{args.start_ds}"
            and experiment_id = "{args.experiment_id}"
            and decision_id is null
            and decision is null
        limit 10
        """,
    )
    print(df)
    print("\n")

    print(
        "Check % of delayed rewards that have a matching MDP id in the decisions table\n"
    )
    df = query_to_df(
        client,
        f"""
        with r as (
            select
                mdp_id
            from `{rewards_table}`
            where date(_partitiontime) >= "{args.start_ds}"
                and experiment_id = "{args.experiment_id}"
                and decision_id is null
                and decision is null
        ), d as (
            select
                distinct mdp_id
            from `{decisions_table}`
            where date(_partitiontime) >= "{args.start_ds}"
                and experiment_id = "{args.experiment_id}"
        )
        select
            sum(case when d.mdp_id is not null then 1 else 0 end) /
            count(*) percent_del_rewards_with_associated_mdp
        from r
        left join d on d.mdp_id = r.mdp_id
        """,
    )
    print(df)
    print("\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--project",
        type=str,
        help="The ID of the GCP project where BigQuery data lives.",
        required=True,
    )
    parser.add_argument(
        "--dataset", type=str, help="Dataset id of tables in BigQuery.", required=True
    )
    parser.add_argument(
        "--creds_path", type=str, help="Path to a GCP credentials file", required=True
    )
    parser.add_argument(
        "--experiment_id",
        type=str,
        help="Name of the experiment in the table.",
        required=True,
    )
    parser.add_argument(
        "--start_ds",
        type=str,
        help="Starting date range of when to check the table",
        required=True,
    )
    args = parser.parse_args()
    main(args)
