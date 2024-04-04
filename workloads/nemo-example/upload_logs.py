from dataclasses import asdict
from datetime import datetime, timedelta
from time import monotonic
from typing import Optional
from uuid import uuid4

import pandas as pd
from google.cloud import bigquery

from dataclasses import dataclass

from google.cloud import bigquery
import google.auth

import os
import pandas as pd
import sys

from tensorflow.python.summary.summary_iterator import summary_iterator

PROJECT_NAME: str = "supercomputer-testing"
_DATASET_NAME: str = "BenchmarkingGallery"
_METADATA_TABLE_NAME: str = "metadata"
_METRICS_TABLE_NAME: str = "metrics"

DATASET_NAME: str = f"{PROJECT_NAME}.{_DATASET_NAME}"


def _full_table_name(name: str) -> str:
    return f"{DATASET_NAME}.{name}"


METRICS_TABLE_NAME: str = _full_table_name(_METRICS_TABLE_NAME)
METADATA_TABLE_NAME: str = _full_table_name(_METADATA_TABLE_NAME)


GSC_TEST_CSV: str = "gs://carterblum/llama2-70b-test-metrics.csv"
LOCAL_TEST_CSV: str = "llama2-70b-test-metrics.csv"


@dataclass
class PreTrainingConfig:
    config_name: str
    chip_name: str
    flops_per_chip: float
    chip_hourly_price: float  # price in Tether :P
    chip_count: int
    region: str
    architecture_shape: str  # not sure how to best parameterize this space
    implementation_name: str
    ml_framework: str  # ['jax', 'pt', 'tf', or, if you're really spicy, 'theano']
    model_name: str
    n_params: float
    precision: str  # not sure how to best parameterize this space
    max_seq_len: int
    batch_size: int
    container_path: str


def clean_name(name: str) -> str:
    return name.replace("/", "__")


metrics_table = bigquery.Table(
    table_ref=METRICS_TABLE_NAME,
    schema=[
        bigquery.SchemaField(
            "run_uid", "STRING", mode="REQUIRED"
        ),  # used for grouping metrics across a single run
        bigquery.SchemaField(
            "config_name", "STRING", mode="REQUIRED"
        ),  # used for joining with config metadata
        bigquery.SchemaField("total_walltime", "FLOAT64", mode="REQUIRED"),
        bigquery.SchemaField("step", "INT64", mode="REQUIRED"),
        bigquery.SchemaField("n_tokens", "INT64", mode="REQUIRED"),
        bigquery.SchemaField("n_samples", "INT64", mode="REQUIRED"),
        bigquery.SchemaField("model_flops", "FLOAT64", mode="REQUIRED"),
        bigquery.SchemaField(
            "wall_time_utc", "DATETIME", mode="REQUIRED"
        ),  # used for debugging and for getting most fresh data
    ],
)

# TODO: this is a pretty hasty implementation of metadata
# a better implementation likely involves a factorized representation, with potentially different tables for
# chip metadata and model metadata
metadata_table = bigquery.Table(
    table_ref=METADATA_TABLE_NAME,
    schema=[
        bigquery.SchemaField("config_name", "STRING", mode="REQUIRED"),
        bigquery.SchemaField("chip_name", "STRING", mode="REQUIRED"),
        bigquery.SchemaField("flops_per_chip", "FLOAT64", mode="REQUIRED"),
        bigquery.SchemaField("chip_hourly_price", "FLOAT64", mode="REQUIRED"),
        bigquery.SchemaField("chip_count", "INT64", mode="REQUIRED"),
        bigquery.SchemaField("region", "STRING", mode="REQUIRED"),
        bigquery.SchemaField("architecture_shape", "STRING", mode="REQUIRED"),
        bigquery.SchemaField("implementation_name", "STRING", mode="REQUIRED"),
        bigquery.SchemaField("ml_framework", "STRING", mode="REQUIRED"),
        bigquery.SchemaField("model_name", "STRING", mode="REQUIRED"),
        bigquery.SchemaField("n_params", "FLOAT64", mode="REQUIRED"),
        bigquery.SchemaField("precision", "STRING", mode="REQUIRED"),
        bigquery.SchemaField("max_seq_len", "INT64", mode="REQUIRED"),
        bigquery.SchemaField("batch_size", "INT64", mode="REQUIRED"),
        bigquery.SchemaField("container_path", "STRING", mode="REQUIRED"),
    ],
)

# gcloud auth application-default login --scopes=https://www.googleapis.com/auth/drive,https://www.googleapis.com/auth/cloud-platform
def _fetch_metadata_from_bigquery(name: str) -> pd.DataFrame:
    credentials, project = google.auth.default(
        scopes=[
            "https://www.googleapis.com/auth/drive",
            "https://www.googleapis.com/auth/cloud-platform",
        ]
    )
    client = bigquery.Client(credentials=credentials, project=project)
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
    df["total_walltime"] = df["train_step_timing in s"]
    df["step"] = df["global_step"]
    df["n_tokens"] = df["global_step"] # df["samples"] * cfg.max_seq_len
    df["model_flops"] = df["global_step"] # df["throughput/flops_per_sec"] * df["time/total"]
    df["n_samples"] = df["global_step"] # df["samples"]
    df["wall_time_utc"] = run_time + (timedelta(seconds=1) * df["total_walltime"])

    df = df.dropna(subset=_cols_matching_table(df, metrics_table, required_only=True))

    assert len(df) != 0, "Drop NA dropped all rows!"

    return df


def upload_df(df: pd.DataFrame, table: bigquery.Table) -> None:
    credentials, project = google.auth.default(
        scopes=[
            "https://www.googleapis.com/auth/drive",
            "https://www.googleapis.com/auth/cloud-platform",
        ]
    )
    client = bigquery.Client(credentials=credentials, project=project)
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

def get_event_file(log_dir):
    for base, dirs, files in os.walk(log_dir):
        for file in files:
            if file.startswith("events"):
                file_path = os.path.join(base, file)
                print (f"Found file: {file_path}")
                return file_path
    return None


def get_values(log_dir):
    required_tags = ["epoch", "global_step", "reduced_train_loss", "train_step_timing in s"]
    data = {tag: [] for tag in required_tags}
    file_path = get_event_file(log_dir)
    
    if file_path is None:
        return {}

    for e in summary_iterator(file_path):
        for v in e.summary.value:
            print (v.tag)
            if v.tag in required_tags:
                data[v.tag].append(v.simple_value)
    
    return data

def dump_as_csv(data):
    df = pd.DataFrame.from_dict(data)
    df.to_csv("logs.csv")

test_training_config = PreTrainingConfig(
    config_name="gpt-5b-2nodes",
    chip_name="h100 nvl",
    flops_per_chip=989.4e12,
    chip_hourly_price=3.928,
    chip_count=16,
    architecture_shape="4x4x4",
    region="us-central1",
    implementation_name="nemo",
    ml_framework="pt",
    model_name="gpt-5b",
    n_params=5e9,
    precision="bf16",
    max_seq_len=4096,
    batch_size=128,
    container_path="NULL",
)

if __name__ == "__main__":
    
    data = get_values(sys.argv[1])
    print ("Print data")
    print (data)
    print ("Done printing data")
    
    df = pd.DataFrame.from_dict(data)
    
    # dump_as_csv(data)
    # df: pd.DataFrame = pd.read_csv(LOCAL_TEST_CSV)
    df = format_litgpt_df(df, test_training_config, datetime.now())
    sync_metadata(test_training_config)
    upload_df(df, metrics_table)
