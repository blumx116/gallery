from dataclasses import asdict, dataclass
from typing import Any

from google.cloud import bigquery

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

    def except_where(self, **kwargs: Any) -> "PreTrainingConfig":
        return PreTrainingConfig(**{**asdict(self), **kwargs})


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
