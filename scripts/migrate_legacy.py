from typing import Optional

import click
import google
from google.cloud import bigquery
from google.oauth2 import service_account
from tqdm import tqdm

from banditml_pkg.banditml.db.bq import TableNotFound
from banditml_pkg.banditml.db.v1 import DecisionTable, RewardTable
from banditml_pkg.banditml.db.v2 import Feedback, FeedbackTable


class Migrator:
    def __init__(
        self,
        client: bigquery.Client,
        project_id: str,
        dataset_id: str,
        dry_run: bool = True,
        create_table_only: bool = False,
    ):
        self.client: bigquery.Client = client
        self.project_id: str = project_id
        self.dataset_id: str = dataset_id
        self.dry_run: bool = dry_run
        self.create_table_only: bool = create_table_only

        try:
            self.reward_table: RewardTable = RewardTable(
                client, project_id=project_id, dataset_id=dataset_id
            )
        except TableNotFound:
            click.echo(
                "Rewards table does not exist, check project and dataset IDs and try again.",
                err=True,
            )
            return

        try:
            self.decision_table: DecisionTable = DecisionTable(
                client, project_id=project_id, dataset_id=dataset_id
            )
        except TableNotFound:
            click.echo(
                "Decisions table does not exist, check project and dataset IDs and try again.",
                err=True,
            )
            return

        self.feedback_table: Optional[FeedbackTable] = None

    def run(self) -> bool:
        try:
            self.feedback_table = FeedbackTable(
                self.client, project_id=self.project_id, dataset_id=self.dataset_id
            )
            click.echo("Feedback table exists already!")
        except TableNotFound:
            should_create = click.prompt(
                "Feedback table does not exist, create?", default=True, type=bool
            )
            if not should_create:
                click.echo("Create feedback table and rerun.", err=True)
                return False
            if not self.dry_run:
                self.feedback_table = FeedbackTable.create(
                    self.client, self.project_id, self.dataset_id
                )
                click.echo("Feedback table created.",)
            else:
                click.echo("Dry run enabled, would have created feedback table.")
                click.echo("Cannot continue dry run without feedback table.")
                return True
        if self.create_table_only:
            click.echo("Data migration disabled, all done!")
            return True
        if self.feedback_table.size() > 0:
            should_overwrite = click.prompt(
                "Feedback table has data, migrate anyway? This might result in duplicated results.",
                type=bool,
                default=True,
                show_default=True,
            )
            if not should_overwrite:
                return False
        self.migrate_decisions()
        self.migrate_rewards()
        return True

    def migrate_rewards(self):
        num_rewards = self.reward_table.size()
        for reward in tqdm(
            self.reward_table.iter_all(), desc="rewards".ljust(10), total=num_rewards
        ):
            feedbacks = Feedback.from_reward("TBD", reward)
            for f in feedbacks:
                if not self.dry_run:
                    self.feedback_table.buffered_write(f.to_dict())
        self.feedback_table.flush()
        click.echo(self.feedback_table.status())

    def migrate_decisions(self):
        num_decisions = self.decision_table.size()
        for decision in tqdm(
            self.decision_table.iter_all(),
            desc="decisions".ljust(10),
            total=num_decisions,
        ):
            feedback = Feedback.from_decision("TBD", decision)
            if not self.dry_run:
                self.feedback_table.buffered_write(feedback.to_dict())
        self.feedback_table.flush()
        click.echo(self.feedback_table.status())

    def print_info(self):
        rewards_size = self.reward_table.size()
        decisions_size = self.decision_table.size()
        click.echo(f"Rewards to migrate: {rewards_size}")
        click.echo(f"Decisions to migrate: {decisions_size}")


@click.command()
@click.argument("gcp-project-id")
@click.argument("gcp-dataset-id")
@click.option(
    "--gcp-creds-file",
    type=click.Path(exists=True),
    help="Path to GCP service account credentials file to use instead of default system GCP credentials.",
)
@click.option(
    "--table-only",
    "--no-data",
    is_flag=True,
    help="Only create the new table, but do not migrate any legacy data",
)
@click.option("-f", "--force", is_flag=True, help="Disables dry-run mode.")
def migrate(gcp_project_id, gcp_dataset_id, gcp_creds_file, table_only, force) -> bool:
    is_dry_run = not force
    try:
        client = create_bq_client(gcp_project_id, gcp_creds_file=gcp_creds_file)
    except google.auth.exceptions.DefaultCredentialsError as e:
        click.echo("❌ GCP credentials invalid:", err=True)
        click.echo(e, err=True)
        click.echo(
            "Migration requires GCP service account credentials with at least BigQuery Data Editor role."
        )
        click.echo(
            "Set the environment variable GOOGLE_APPLICATION_CREDENTIALS to "
            "the path of your credentials or specify the --gcp-creds-file CLI option."
        )
        return False
    if is_dry_run:
        click.echo("*** DRY RUN (run with -f/--force to write data) ***")
    return Migrator(
        client,
        gcp_project_id,
        gcp_dataset_id,
        dry_run=is_dry_run,
        create_table_only=table_only,
    ).run()


def create_bq_client(project_id, gcp_creds_file=None):
    credentials = None
    if gcp_creds_file:
        credentials = service_account.Credentials.from_service_account_file(
            gcp_creds_file
        )

    return bigquery.Client(project=project_id, credentials=credentials)


if __name__ == "__main__":
    exit(0 if migrate() else 1)
