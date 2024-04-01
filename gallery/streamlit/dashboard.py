from typing import Optional, Union

import pandas as pd
import streamlit as st
from google.cloud import bigquery

from gallery.constants import metadata_table, metrics_table
from gallery.streamlit.utils import human_readable_bignum

st.title("Benchmarking Gallery Dashboard")

Primitive = Union[str, float, int]


@st.cache_resource
def client() -> bigquery.Client:
    return bigquery.Client()


def _format_constraint(field: str, value: Primitive) -> str:
    if isinstance(value, str):
        value = f"'{value}'"  # TODO: this is probably not safe if there are any unescaped quotes in the value...
        # not sure how likely that is
    return f"{field} = {value}"


def _format_constraints(**constraints: Primitive) -> str:
    constraints_command: str = "\n AND ".join(
        [_format_constraint(field, value) for field, value in constraints.items()]
    )
    if constraints_command:
        constraints_command = "\nWHERE\n" + constraints_command
    return constraints_command


def _partition_command(*group_by_elems: list[str]) -> str:
    if not group_by_elems:
        # create trivial partition function that returns 1
        return ""
    else:
        # TODO: in progress
        pass


def query_unique(
    table: bigquery.Table, target: str, **constraints: Primitive
) -> list[Primitive]:
    query: str = (
        f"""
        SELECT DISTINCT {target}
        FROM {str(table)}
    """
        + _format_constraints(**constraints)
    )
    result: pd.DataFrame = client().query_and_wait(query).to_dataframe()
    return result[target].to_list()


@st.cache_data
def available_model_names() -> list[str]:
    return query_unique(metadata_table, "model_name")  # type: ignore


@st.cache_data
def available_model_params(model_name: str) -> list[float]:
    return query_unique(metadata_table, "n_params", model_name=model_name)  # type: ignore


@st.cache_data
def available_batch_sizes(model_name: str, model_params: float) -> list[int]:
    return query_unique(metadata_table, "batch_size", model_name=model_name, n_params=model_params)  # type: ignore


@st.cache_data
def available_sequence_lengths(model_name: str, model_params: float) -> list[int]:
    return query_unique(metadata_table, "max_seq_len", model_name=model_name, n_params=model_params)  # type: ignore


all_run_uid_command: str = (
    f"SELECT run_uid, config_name, MIN(wall_time_utc) AS start_time FROM {str(metrics_table)} GROUP BY run_uid, config_name"
)
# this query gets us each run_uid, with its config_name and its start time


@st.cache_data
def get_most_recent_runs(
    constraints: dict[str, Primitive], groupby: list[str]
) -> list[str]:
    # returns the run_uid for the most recent run where runs are deduped by the columns in groupby
    # and filtered by the constraints in constraints
    runs_with_metadata_command: str = f"""
        SELECT m.run_uid, m.config_name, m.start_time, md.*
        FROM ({all_run_uid_command}) m
        JOIN {str(metadata_table)} md ON m.config_name = md.config_name
    """
    # This command returns a table with (run_uid, start_time, config_name, **metadata) for each run

    filtered_runs_command: str = (
        f"""
        SELECT * 
        FROM ({runs_with_metadata_command}) 
    """
        + _format_constraints(**constraints)
    )
    # This command outputs the same thing as above, but filtered by constraints

    full_command: str = f""" 
        WITH runs_with_metadata AS ({filtered_runs_command})
        SELECT run_uid 
        FROM (
            SELECT run_uid, ROW_NUMBER() OVER ({_partition_command(groupby)}) as row_num
        )
        WHERE row_num = 1
    """

    st.write(filtered_runs)


get_most_recent_runs(constraints={"chip_name": "v5e"}, groupby=[])


with st.expander("Chart Configuration"):
    model: Optional[str] = st.radio("Model Name", options=available_model_names())

    assert model is not None
    model_params: Optional[float] = st.radio(
        "Model Params",
        options=available_model_params(model),
        format_func=human_readable_bignum,
    )
    assert model_params is not None
    batch_size: Optional[int] = st.radio(
        "Batch Size", options=available_batch_sizes(model, model_params)
    )
    max_seq_len: Optional[int] = st.radio(
        "Sequence Length", options=available_sequence_lengths(model, model_params)
    )

y_axis: Optional[str] = st.radio(
    "Y Axis", options=["MFU", "Step Time", "Tokens/Sec"], index=0
)
