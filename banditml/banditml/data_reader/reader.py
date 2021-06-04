from typing import Dict, Optional

import pandas as pd
from google.cloud import bigquery
from google.oauth2 import service_account

from ..utils.utils import get_logger

logger = get_logger(__name__)


class BigQueryReader:
    def __init__(self, credential_path: str):
        credentials = service_account.Credentials.from_service_account_file(
            filename=credential_path,
            scopes=["https://www.googleapis.com/auth/cloud-platform"],
        )
        project_id = credentials.project_id
        self.client = bigquery.Client(credentials=credentials, project=project_id)

    def get_dataset(self, query: str) -> pd.DataFrame:
        """Execute adhoc / one-off blocking query."""
        return self.client.query(query).result().to_dataframe()
