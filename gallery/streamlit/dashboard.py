from typing import Optional

import pandas as pd
import plotly.express as px
import streamlit as st
from pandas.api.types import is_datetime64_any_dtype

from gallery.streamlit.elements import filter_config
from gallery.streamlit.queries import (
    Primitive,
    filtered_runs,
    get_data_for_run_uid,
    get_most_recent_run_uids,
)
from gallery.streamlit.utils import dict_diff, pretty_dict

st.title("Benchmarking Dashboard")

benchmarking_tab, regression_test_tab, experiment_comparison_tab = st.tabs(
    ["Benchmarking", "Regression Test", "Experiment Comparison"]
)


def get_last_timestep(df: pd.DataFrame) -> pd.DataFrame:
    return df.loc[df.groupby("run_uid")["total_walltime"].idxmax()]


def calculate_derived_metrics(df: pd.DataFrame) -> pd.DataFrame:
    df["MFU"] = df["model_flops"] / (
        df["flops_per_chip"] * df["chip_count"] * df["total_walltime"]
    )
    df["Step Time"] = df["total_walltime"] / df["step"]
    df["Tokens/Sec"] = df["n_tokens"] / df["total_walltime"]

    df["Architecture"] = df["chip_name"] + " " + df["chip_count"].astype(str)
    df["Perf/$"] = df["Tokens/Sec"] / (df["chip_count"] * df["chip_hourly_price"])
    df["Elapsed Time"] = pd.to_datetime(df["total_walltime"], unit="s")
    return df


def calculate_stepwise_derived_metrics(df: pd.DataFrame) -> pd.DataFrame:
    # TODO: this is just a duplicate of calculate_derivative_metrics except we wrap a couple columns
    # that contain cumulative values with __bystep
    # the last 3 columns are literally identical

    def __bystep(col: str) -> pd.Series:
        # group by run_uid so that the first value of one run isn't compared against the last value
        # of the run before it
        return df.groupby(["run_uid"])[col].diff()  # type: ignore

    df["MFU"] = __bystep("model_flops") / (
        df["flops_per_chip"] * df["chip_count"] * __bystep("total_walltime")
    )
    df["Step Time"] = __bystep("total_walltime") / __bystep("step")
    df["Tokens/Sec"] = __bystep("n_tokens") / __bystep("total_walltime")

    df["Perf/$"] = df["Tokens/Sec"] / (df["chip_count"] * df["chip_hourly_price"])
    df["Elapsed Time"] = pd.to_datetime(df["total_walltime"], unit="s")
    return df


def accelerator_comparison(widget_key_prefix: str = "accelerator_comparison"):
    st.header("Performance by Accelerator")

    configs, charts = st.columns([1, 2])

    with configs:
        with st.expander("Data Filters", expanded=True):
            filter_constraints: dict[str, Primitive] = filter_config(
                [
                    ("model_name", "Model Name"),
                    ("n_params", "Model Param Count"),
                    ("batch_size", "Batch Size"),
                    ("max_seq_len", "Sequence Length"),
                ],
                widget_key_prefix=widget_key_prefix,
            )

        y_axis: Optional[str] = st.radio(
            "Y Axis",
            options=["MFU", "Step Time", "Tokens/Sec", "Perf/$"],
            key=widget_key_prefix + "_y_axis",
        )

    run_uids: list[str] = get_most_recent_run_uids(
        constraints=filter_constraints,
        groupby=["chip_name", "chip_count"],
    )

    df: pd.DataFrame = get_data_for_run_uid(run_uids)

    df_tlast: pd.DataFrame = get_last_timestep(df)
    # df_tlast contains the last timestep for each run

    df_tlast = calculate_derived_metrics(df_tlast)

    with charts:
        fig = px.bar(
            df_tlast,
            x="Architecture",
            y=y_axis,
            title=f"Performance ({y_axis}) by Architecture",
        )

        if y_axis == "MFU":
            fig.update_yaxes(tickformat=".1%")

        st.plotly_chart(fig)


def single_experiment_run(widget_key_prefix: str = "single_experiment_run"):
    st.header("Performance by Epoch")
    configs, charts = st.columns([1, 2])
    with configs:
        with st.expander("Data Filters", expanded=True):
            filter_constraints: dict[str, Primitive] = filter_config(
                [
                    ("model_name", "Model Name"),
                    ("n_params", "Model Param Count"),
                    ("batch_size", "Batch Size"),
                    ("max_seq_len", "Sequence Length"),
                    ("chip_name", "Accelerator Type"),
                    ("chip_count", "Num Accelerators"),
                    ("implementation_name", "Implementation"),
                ],
                widget_key_prefix=widget_key_prefix,
            )

        x_axis: Optional[str] = st.radio(
            "X Axis",
            options=["Elapsed Time", "step"],
            key=widget_key_prefix + "_x_axis",
        )

        y_axis: Optional[str] = st.radio(
            "Y Axis",
            options=["MFU", "Step Time", "Tokens/Sec", "Perf/$"],
            key=widget_key_prefix + "_y_axis",
        )

    run_uids: list[str] = get_most_recent_run_uids(
        constraints=filter_constraints, groupby=[]
    )
    df: pd.DataFrame = get_data_for_run_uid(run_uids)
    df = calculate_stepwise_derived_metrics(df)

    with charts:
        fig = px.scatter(df, x=x_axis, y=y_axis)

        if is_datetime64_any_dtype(df[x_axis]):
            fig.update_xaxes(tickformat="%H:%M")

        if y_axis == "MFU":
            fig.update_yaxes(tickformat=".1%")

        st.plotly_chart(fig)

    st.header("Performance vs Historical Runs")

    dfhist: pd.DataFrame = calculate_derived_metrics(
        get_last_timestep(get_data_for_run_uid(filtered_runs(filter_constraints)))
    )

    dfhist["Finish Time (UTC)"] = pd.to_datetime(dfhist["wall_time_utc"])

    explanation, chart2 = st.columns([1, 2])

    with explanation:
        st.write("Information shown here is for the same experiment as above.")

    with chart2:
        fig = px.scatter(dfhist, x="Finish Time (UTC)", y=y_axis)
        fig.update_xaxes(tickformat="%H:%M")

        if y_axis == "MFU":
            fig.update_yaxes(tickformat=".1%")

        st.plotly_chart(fig)


def two_experiments_side_by_side(widget_key_prefix="two_experiments_side_by_side"):
    config1, config2, shared, chart = st.columns([1, 1, 1, 2])

    with config1:
        filter_constraints1: dict[str, Primitive] = filter_config(
            [
                ("model_name", "Model Name"),
                ("n_params", "Model Param Count"),
                ("batch_size", "Batch Size"),
                ("max_seq_len", "Sequence Length"),
                ("chip_name", "Accelerator Type"),
                ("chip_count", "Num Accelerators"),
                ("implementation_name", "Implementation"),
            ],
            widget_key_prefix=widget_key_prefix + "1",
        )

    with config2:
        filter_constraints2: dict[str, Primitive] = filter_config(
            [
                ("model_name", "Model Name"),
                ("n_params", "Model Param Count"),
                ("batch_size", "Batch Size"),
                ("max_seq_len", "Sequence Length"),
                ("chip_name", "Accelerator Type"),
                ("chip_count", "Num Accelerators"),
                ("implementation_name", "Implementation"),
            ],
            widget_key_prefix=widget_key_prefix + "2",
        )

    with shared:
        y_axis: Optional[str] = st.radio(
            "Y Axis",
            options=["MFU", "Step Time", "Tokens/Sec", "Perf/$"],
            key=widget_key_prefix + "_y_axis",
        )

    df1 = calculate_derived_metrics(
        get_last_timestep(
            get_data_for_run_uid(
                get_most_recent_run_uids(filter_constraints1, groupby=[])
            )
        )
    )
    df2 = calculate_derived_metrics(
        get_last_timestep(
            get_data_for_run_uid(
                get_most_recent_run_uids(filter_constraints2, groupby=[])
            )
        )
    )

    with chart:
        if filter_constraints1 == filter_constraints2:
            st.write(
                "Chart will display here after two different configurations are chosen"
            )
        else:
            plot_df: pd.DataFrame = pd.DataFrame.from_records(
                [
                    {
                        "name": pretty_dict(
                            dict_diff(filter_constraints1, filter_constraints2)
                        ),
                        **df1.to_dict("records")[0],
                    },
                    {
                        "name": pretty_dict(
                            dict_diff(filter_constraints2, filter_constraints1)
                        ),
                        **df2.to_dict("records")[0],
                    },
                ]
            )

            fig = px.bar(plot_df, x="name", y=y_axis)

            if y_axis == "MFU":
                fig.update_yaxes(tickformat=".1%")

            st.plotly_chart(fig)


if st.button("Refresh Data"):
    st.cache_data.clear()

with benchmarking_tab:
    accelerator_comparison()

with regression_test_tab:
    single_experiment_run()

with experiment_comparison_tab:
    two_experiments_side_by_side()
