"""BQ contains base classes and utilities for working with BigQuery as a BanditML data source."""

from typing import Any, ClassVar, Dict, Iterator, List, Sequence

import google
from google.cloud import bigquery

from typing_extensions import TypedDict


class BQError(TypedDict):
    index: int
    errors: List[str]


class Table:
    """A base class for OO BQ table interactions.

    Subclasses must implement `mapper`, and may override class attribute `name` with the table_id, and
    `partition_field`.
    """

    DEFAULT_BUFFER_SIZE = 10000
    SCHEMA: ClassVar[List[bigquery.SchemaField]] = None

    name: ClassVar[str] = None
    partition_field: ClassVar[str] = "timestamp"

    bq_table: bigquery.Table

    @classmethod
    def create(
        cls,
        client: bigquery.Client,
        project_id: str,
        dataset_id: str,
        schema: List[bigquery.SchemaField] = None,
        partition: bool = False,
    ):
        if cls.name is None:
            raise NotImplementedError(
                "class attribute `name` must be set in order to create table"
            )
        bq_table = bigquery.Table(
            _fulltable(project_id, dataset_id, cls.name),
            schema=schema,
        )
        bq_table.time_partitioning = bigquery.TimePartitioning(
            type_=bigquery.TimePartitioningType.DAY,
            field=cls.partition_field,
        )
        bq_table = client.create_table(bq_table)

        return cls(client, bq_table=bq_table)

    def __init__(
        self,
        client: bigquery.Client,
        bq_table: bigquery.Table = None,
        project_id: str = None,
        dataset_id: str = None,
    ):
        full_path_ok: bool = (
            project_id is not None and dataset_id is not None and self.name is not None
        )
        if bq_table is None and not full_path_ok:
            raise ValueError(
                "bq_table or project_id, dataset_id, and class variable 'name' are required arguments"
            )
        self.client: bigquery.Client = client
        if bq_table:
            self.bq_table = bq_table
            if self.name is not None and bq_table.table_id != self.name:
                raise ValueError(
                    f"provided bq_table does not match name, {bq_table.table_id} != {self.name}"
                )
        else:
            self.bq_table = get_table(client, project_id, dataset_id, self.name)
        self.full_table_name: str = self.bq_table.full_table_id.replace(":", ".")
        self.buffer: List = []
        self.total_written: int = 0
        self.total_errors: int = 0
        self.bad_records: List = []

    def size(self) -> int:
        since_beginning_of_time = self._partition_field_clause()
        result = self.client.query(
            f"SELECT COUNT(1) as count FROM `{self.full_table_name}` WHERE {since_beginning_of_time}"
        ).result()
        for row in result:
            return row.values()[0]

    def iter_all(self) -> Iterator[Any]:
        """Returns an iterator over all records in a table since 2000, the beginning of all time."""
        since_beginning_of_time = self._partition_field_clause()
        query = (
            f"SELECT * FROM `{self.full_table_name}` WHERE {since_beginning_of_time}"
        )
        for row in self.client.query(query).result():
            # leveraging bigquery's built-in pagination instead of reimplementing our own.
            yield self.mapper(row)

    def all(self) -> Sequence[Any]:
        """Returns all records in a table since 2000, the beginning of all time."""
        return [r for r in self.iter_all()]

    def write_all(self, records) -> Sequence[BQError]:
        """Immediately write all records."""
        if not records:
            return []
        errors = self.client.insert_rows(
            table=self.bq_table,
            rows=records,
            skip_invalid_rows=False,
            ignore_unknown_values=False,
        )
        return self._filter_errors(errors)

    def buffered_write(self, record: Dict):
        """Buffer write a record.

        Buffer will be flushed when size is >= DEFAULT_BUFFER_SIZE or when `flush` is called.
        """
        if record:
            self.buffer.append(record)
        if len(self.buffer) >= self.DEFAULT_BUFFER_SIZE:
            self._write_buffer()

    def flush(self):
        """Flushes the write buffer."""
        self._write_buffer()

    def mapper(self, row):
        """(Abstract) Maps a BQ row to a BanditML model."""
        raise NotImplementedError()

    def status(self):
        s = (
            f"wrote {self.total_written} "
            f"with {self.total_errors} errors "
            f"to {self.name}."
        )
        if self.total_errors > 0:
            s += f"\nsample of bad records:\n{self.bad_records[-5:]}"
        return s

    @staticmethod
    def _filter_errors(errors):
        filtered_errors = []
        for e1 in errors:
            for e2 in e1["errors"]:
                if e2.get("reason", None) != "stopped":
                    filtered_errors.append(e1)
        return filtered_errors

    def _partition_field_clause(self):
        return (
            f"""DATE({self.partition_field}) > "2000-01-01" """
            if self.partition_field
            else ""
        )

    def _write_buffer(self):
        num_records = len(self.buffer)
        errors = self.write_all(self.buffer)
        self.bad_records.extend(
            {"record": self.buffer[error["index"]], "errors": error["errors"]}
            for error in errors
        )
        self.buffer = []
        num_errors = len(errors)
        num_successes = num_records - num_errors
        self.total_written += num_successes
        self.total_errors += num_errors


def get_table(client, project_id, dataset_id, name) -> bigquery.Table:
    fulltable = _fulltable(project_id, dataset_id, name)
    try:
        return client.get_table(fulltable)
    except google.api_core.exceptions.NotFound:
        raise TableNotFound(fulltable)


class TableNotFound(Exception):
    def __init__(self, fulltable):
        self.fulltable = fulltable

    def __str__(self):
        return f"Table {self.fulltable} does not exist"


def _fulltable(project_id, dataset_id, table):
    return f"{project_id}.{dataset_id}.{table}"
