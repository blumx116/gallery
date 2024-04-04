from dataclasses import asdict
from datetime import datetime, timedelta
from time import monotonic
from typing import Optional
from uuid import uuid4

import pandas as pd
from google.cloud import bigquery

import argparse

from gallery.constants import (
    LOCAL_TEST_CSV,
    PROJECT_NAME,
    PreTrainingConfig,
    metadata_table,
    metrics_table,
)

test_training_config = PreTrainingConfig(
    config_name="llama2-70b-h100 nvlx64",
    chip_name="h100 nvl",
    flops_per_chip=989.4e12,
    chip_hourly_price=3.928,
    chip_count=64,
    architecture_shape="4x4x4",
    region="us-central1",
    implementation_name="litgpt",
    ml_framework="pt",
    model_name="llama-2",
    n_params=70e9,
    precision="bf16",
    max_seq_len=4096,
    batch_size=6,
    container_path="NULL",
)


def _fetch_metadata_from_bigquery(name: str) -> pd.DataFrame:
    client = bigquery.Client()
    query: str = f"""
        SELECT * 
        FROM {str(metadata_table)}
        WHERE config_name = '{name}'
    """
    job = client.query(query)
    result = job.result()

    return result.to_dataframe()


def sync_metadata(cfg: PreTrainingConfig, overwrite: bool = False) -> None:
    cfg_df = pd.DataFrame.from_records([asdict(cfg)])

    if not overwrite:
        data: pd.DataFrame = _fetch_metadata_from_bigquery(cfg.config_name)
        if len(data):
            # computing df equality this way b/c dtypes might not be the same
            # so .equals() doesn't behave as expected
            assert (cfg_df == data).all(
                axis=None
            ), f"provided configuration {cfg.config_name} doesn't match database entry:\n\nProvided:\n{cfg_df}\n\nFetched:\n{data}"

            return None

    _upload_df(cfg_df, metadata_table)


def _cols_matching_table(
    df: pd.DataFrame, table: bigquery.Table, required_only: bool = False
) -> list[str]:
    fields_to_select: list[bigquery.SchemaField] = table.schema
    if required_only:
        fields_to_select = [
            field for field in fields_to_select if field.mode == "REQUIRED"
        ]

    overlapping_columns: list[str] = [
        field.name for field in fields_to_select if field.name in df.columns
    ]

    for field in table.schema:
        if field.mode == "REQUIRED":
            assert (
                field.name in overlapping_columns
            ), f"Missing required column: {field}"

    assert len(overlapping_columns), "Filtering columns by schema removed all columns!"

    return overlapping_columns


def format_litgpt_df(
    df: pd.DataFrame, cfg: PreTrainingConfig, run_time: datetime
) -> pd.DataFrame:
    df["run_uid"] = str(uuid4())
    df["config_name"] = cfg.config_name
    df["total_walltime"] = df["time/total"]
    df["step"] = df["step"]
    df["n_tokens"] = df["samples"] * cfg.max_seq_len
    df["model_flops"] = df["throughput/flops_per_sec"] * df["time/total"]
    df["n_samples"] = df["samples"]
    df["wall_time_utc"] = run_time + (timedelta(seconds=1) * df["time/total"])

    df = df.dropna(subset=_cols_matching_table(df, metrics_table, required_only=True))

    assert len(df) != 0, "Drop NA dropped all rows!"

    return df


def upload_df(df: pd.DataFrame, table: bigquery.Table) -> None:
    client = bigquery.Client(project=PROJECT_NAME)
    job_config = bigquery.LoadJobConfig(
        schema=table.schema,
    )
    df = df[_cols_matching_table(df, table, required_only=False)]  # type: ignore
    job = client.load_table_from_dataframe(
        df,
        table,
        job_config=job_config,
    )
    job.result()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parse metrics.csv and upload to Big Query")
    parser.add_argument("filepath", help="The path of metrics.csv to process")

    args = parser.parse_args()

    df: pd.DataFrame = pd.read_csv(args.filepath)
    df = format_litgpt_df(df, test_training_config, datetime.now())
    sync_metadata(test_training_config)
    upload_df(df, metrics_table)
