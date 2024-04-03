from typing import Optional, Union

import pandas as pd
import streamlit as st
from google.api_core.exceptions import BadRequest
from google.cloud import bigquery

from gallery.constants import metadata_table, metrics_table

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
    result: str = "ORDER BY start_time DESC"
    if len(group_by_elems) > 0:
        casts: list[str] = [f"CAST({elem} AS STRING)" for elem in group_by_elems]
        # this is probably bad practice, but one of the things I'm currently allowing people to partition by is
        # n_params and you can't partition by floats
        # b/c I expect the table of just filtered run_uids to be fairly small at this point, I'm just casting
        # everything to string rather than figure out what I need to cast
        result = "PARTITION BY " + ",".join(casts) + " " + result
    return result


@st.cache_data
def query_unique(
    _table: bigquery.Table, target: str, **constraints: Primitive
) -> list[Primitive]:
    """
    _table is so named b/c the leading underscore tells streamlit not to hash it
    """
    query: str = (
        f"""
        SELECT DISTINCT {target}
        FROM {str(_table)}
    """
        + _format_constraints(**constraints)
    )

    try:
        result: pd.DataFrame = client().query_and_wait(query).to_dataframe()
        return result[target].to_list()
    except BadRequest:
        st.text(query)
        raise


all_run_uid_command: str = (
    f"SELECT run_uid, config_name, MIN(wall_time_utc) AS start_time FROM {str(metrics_table)} GROUP BY run_uid, config_name"
)
# this query gets us each run_uid, with its config_name and its start time


def filtered_runs_command(constraints: dict[str, Primitive]) -> str:
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
    return filtered_runs_command


@st.cache_data
def get_most_recent_run_uids(
    constraints: dict[str, Primitive], groupby: list[str]
) -> list[str]:
    # returns the run_uid for the most recent run where runs are deduped by the columns in groupby
    # and filtered by the constraints in constraints

    full_command: str = f""" 
        WITH runs_with_metadata AS ({filtered_runs_command(constraints)})
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
    try:
        return client().query_and_wait(full_command).to_dataframe()["run_uid"].to_list()
    except BadRequest:
        st.text(full_command)
        raise


@st.cache_data
def filtered_runs(constraints: dict[str, Primitive]) -> list[str]:
    query: str = filtered_runs_command(constraints)
    try:
        return client().query_and_wait(query).to_dataframe()["run_uid"].to_list()
    except BadRequest:
        st.text(query)
        raise


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
    try:
        return client().query_and_wait(query).to_dataframe()
    except BadRequest:
        st.text(query)
        raise
