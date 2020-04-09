import pandas as pd
from google.cloud import bigquery
from google.oauth2 import service_account
from retry import retry
from typing import Optional

from utils.utils import get_logger


logger = get_logger(__name__)


class BigQueryReader:
    def __init__(
        self,
        credential_path: Optional[str],
        decisions_table_name: str,
        rewards_table_name: str,
        decisions_ds_start: str,
        decisions_ds_end: str,
        rewards_ds_end: str,
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
        self.decisions_table_name = decisions_table_name
        self.rewards_table_name = rewards_table_name
        self.decisions_ds_start = decisions_ds_start
        self.decisions_ds_end = decisions_ds_end
        self.rewards_ds_end = rewards_ds_end
        self.experiment_id = experiment_id
        self.training_query = self._build_query()

    def _build_query(self):

        return f"""
        with immediate_r as (
            select
                decision_id,
                decision,
                mdp_id,
                metrics,
                experiment_id
            from `{self.rewards_table_name}`
            where DATE(_PARTITIONTIME)
                between "{self.decisions_ds_start}" and "{self.rewards_ds_end}"
                and experiment_id = "{self.experiment_id}"
        ),
        delayed_r as (
            select
                *
            from immediate_r
            where decision_id is null
              and decision is null
        )
        select
          d.context,
          d.decision,
          immediate_r.metrics immediate_reward,
          delayed_r.metrics delayed_reward
        from `{self.decisions_table_name}` d
            left join immediate_r
              on immediate_r.decision_id = d.decision_id
              and immediate_r.decision = d.decision
              and immediate_r.experiment_id = d.experiment_id
              and immediate_r.mdp_id = d.mdp_id
            left join delayed_r
              on delayed_r.experiment_id = d.experiment_id
              and delayed_r.mdp_id = d.mdp_id
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
