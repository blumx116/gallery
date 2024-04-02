from typing import Optional, Union

import pandas as pd
import plotly.express as px
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


def _partition_command(*group_by_elems: str) -> str:
    if not group_by_elems:
        # create trivial partition function that returns 1
        return ""
    else:
        casts: list[str] = [f"CAST({elem} AS STRING)" for elem in group_by_elems]
        # this is probably bad practice, but one of the things I'm currently allowing people to partition by is
        # n_params and you can't partition by floats
        # b/c I expect the table of just filtered run_uids to be fairly small at this point, I'm just casting
        # everything to string rather than figure out what I need to cast
        return "PARTITION BY " + ",".join(casts) + " ORDER BY start_time DESC"


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
def get_most_recent_run_uids(
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
            SELECT run_uid, ROW_NUMBER() OVER ({_partition_command(*groupby)}) AS row_num FROM runs_with_metadata
        )
        WHERE row_num = 1
    """
    # the additions to the command here filter to only show run_uids where it has a later start time
    # than other runs with the same values in the groupby columns
    # e.g. if groupby=["model_name", "n_params"], it will only show the most recent run_uid
    # for each mode-Name, n_params pair
    return client().query_and_wait(full_command).to_dataframe()["run_uid"].to_list()


@st.cache_data
def get_data_for_run_uid(run_uids: list[str]) -> pd.DataFrame:
    formatted_uids: str = ",".join([f"'{uid}'" for uid in run_uids])
    query: str = f"""
        SELECT m.*, md.*
        FROM (
            SELECT * FROM {str(metrics_table)} WHERE run_uid in ({formatted_uids})
        ) m
        JOIN {str(metadata_table)} md
        ON m.config_name = md.config_name
    """
    return client().query_and_wait(query).to_dataframe()


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

run_uids: list[str] = get_most_recent_run_uids(
    constraints={
        "model_name": model,
        "n_params": model_params,
        "batch_size": batch_size,
        "max_seq_len": max_seq_len,
    },
    groupby=["chip_name", "chip_count"],
)

df: pd.DataFrame = get_data_for_run_uid(run_uids)


df_tlast: pd.DataFrame = df.loc[df.groupby("run_uid")["total_walltime"].idxmax()]
# df_tlast contains the last timestep for each run


def calculate_derivative_metrics(df: pd.DataFrame) -> pd.DataFrame:
    df["MFU"] = df["model_flops"] / (
        df["flops_per_chip"] * df["chip_count"] * df["total_walltime"]
    )
    df["Step Time"] = df["total_walltime"] / df["step"]
    df["Tokens/Sec"] = df["n_tokens"] / df["total_walltime"]

    df["Architecture"] = df["chip_name"] + " " + df["chip_count"].astype(str)
    return df


df_tlast = calculate_derivative_metrics(df_tlast)


st.plotly_chart(
    px.bar(df_tlast, x="Architecture", y=y_axis, title="Performance by Architecture")
)
