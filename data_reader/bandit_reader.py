import pandas as pd
from google.cloud import bigquery
from google.oauth2 import service_account
from retry import retry
from typing import Dict, Optional

from utils.utils import get_logger


logger = get_logger(__name__)


class BigQueryReader:
    def __init__(
        self,
        credential_path: Optional[str],
        bq_project: str,
        bq_dataset: str,
        decisions_ds_start: str,
        decisions_ds_end: str,
        rewards_ds_end: str,
        reward_function: Dict[str, float],
        experiment_id: str,
    ):
        if credential_path:
            credentials = service_account.Credentials.from_service_account_file(
                filename=credential_path,
                scopes=["https://www.googleapis.com/auth/cloud-platform"],
            )
            project_id = credentials.project_id
        else:
            credentials = None
            project_id = None

        self.client = bigquery.Client(credentials=credentials, project=project_id)
        self.decisions_table_name = f"{bq_project}.{bq_dataset}.decisions"
        self.rewards_table_name = f"{bq_project}.{bq_dataset}.rewards"
        self.decisions_ds_start = decisions_ds_start
        self.decisions_ds_end = decisions_ds_end
        self.rewards_ds_end = rewards_ds_end
        self.experiment_id = experiment_id
        self.training_query = self._build_query(reward_function)

    def _build_query(self, reward_function):
        tmp_funcs = '''
        CREATE TEMP FUNCTION getJsonKeys(jsonStr STRING)
        RETURNS Array<String>
        LANGUAGE js AS """
        return Object.keys(JSON.parse(jsonStr));
        """;

        CREATE TEMP FUNCTION getJsonKey(jsonStr STRING, key STRING)
        RETURNS STRING
        LANGUAGE js AS """
        return JSON.stringify(JSON.parse(jsonStr)[key]);
        """;
        '''

        immediate_rewards = ""
        delayed_rewards = ""
        for reward_name, weight in reward_function.items():
            immediate_rewards += f"(coalesce(cast(json_extract(metrics, '$.{reward_name}') as numeric), 0) * {weight}) + "
            delayed_rewards += f"(coalesce(cast(json_extract(getJsonKey(metrics, decision), '$.{reward_name}') as numeric), 0) * {weight}) + "

        # delete trailing plus sign and space + name this row
        immediate_rewards = immediate_rewards[:-2] + "as immediate_reward"
        delayed_rewards = delayed_rewards[:-2] + "as delayed_reward"

        return f"""
        {tmp_funcs}

        with immediate_r as (
            select
                mdp_id,
                decision_id,
                decision,
                {immediate_rewards}
            from `{self.rewards_table_name}`
            where DATE(_PARTITIONTIME)
                between "{self.decisions_ds_start}" and "{self.rewards_ds_end}"
                and experiment_id = "{self.experiment_id}"
                and decision_id is not null
                and decision is not null
        ), grouped_immediate_reward as (
            select
                mdp_id,
                decision_id,
                decision,
                sum(immediate_reward) immediate_reward
            from immediate_r
            group by 1, 2, 3
        ), delayed_r as (
            select
                mdp_id,
                metrics,
                getJsonKeys(metrics) decision_keys
            from `{self.rewards_table_name}`
            where DATE(_PARTITIONTIME)
            between "{self.decisions_ds_start}" and "{self.rewards_ds_end}"
            and experiment_id = "{self.experiment_id}"
            and decision_id is null
            and decision is null
        ), flattened_delayed_r as (
            select
                mdp_id,
                decision,
                {delayed_rewards}
            from delayed_r,
            unnest(decision_keys) as decision
        ), grouped_delayed_r as (
            select
                mdp_id,
                decision,
                sum(delayed_reward) delayed_reward
            from flattened_delayed_r
            group by 1, 2
        )
        select
            d.context,
            d.decision,
            coalesce(gir.immediate_reward, 0) + coalesce(gdr.delayed_reward, 0) reward
        from `{self.decisions_table_name}` d
            left join grouped_immediate_reward gir
                on gir.mdp_id = d.mdp_id
                and gir.decision_id = d.decision_id
                and gir.decision = d.decision
            left join grouped_delayed_r gdr
                on gdr.mdp_id = d.mdp_id
                and gdr.decision = d.decision
        where date(d._PARTITIONTIME)
            between "{self.decisions_ds_start}" and "{self.decisions_ds_end}"
            and d.experiment_id = "{self.experiment_id}"
        """

    @retry(tries=3, delay=1, backoff=2, logger=logger)
    def adhoc_query(self, query: str) -> pd.DataFrame:
        """Execute adhoc / one-off blocking query."""
        return self.client.query(query).result().to_dataframe()

    def get_training_data(self) -> pd.DataFrame:
        """Execute adhoc / one-off blocking query."""
        return self.adhoc_query(self.training_query)
