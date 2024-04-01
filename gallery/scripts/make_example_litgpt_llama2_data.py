"""
NOTE: this script is intended to create silly sample data for debugging purposes.

The data from it should ABSOLUTELY be removed from the database before showing it to a consumer.

Normally, I would suggest that we have a dev database and a prod database, but this POC isn't intended to be longlived and we will be changing our entire stack shortly, so I don't think it's worth the effort.
"""

from dataclasses import asdict
from datetime import datetime, timedelta
from uuid import uuid4

import pandas as pd

from gallery.constants import PreTrainingConfig, metadata_table, metrics_table
from gallery.upload_csv import (
    _format_litgpt_df,
    _sync_metadata,
    _upload_df,
    test_training_config,
)

METRICS_TO_FABRICATE: list[str] = ["n_tokens", "n_samples", "model_flops"]


def old_data_that_should_be_ignored(
    dfbase: pd.DataFrame, cfgbase: PreTrainingConfig
) -> tuple[pd.DataFrame, PreTrainingConfig]:
    """
    This dataset has the exact same config as the base
    but it should be ignored because it was run a day earlier.

    You can recognize it by the fact that all of its metrics are -1
    """

    dfnew: pd.DataFrame = dfbase.copy()
    dfnew["run_uid"] = str(uuid4())
    for column in METRICS_TO_FABRICATE:
        dfnew[column] = -1
    dfnew["wall_time_utc"] = dfbase["wall_time_utc"] - timedelta(days=1)

    return dfnew, cfgbase


def data_on_v5(
    dfbase: pd.DataFrame, cfgbase: PreTrainingConfig
) -> tuple[pd.DataFrame, PreTrainingConfig]:
    """
    Same as the base data, except that it lists that it was run on v5
    instead of h100s

    Because v5's are so cool, all of the goodput measures are x2 and the
    price per chip is halved! I'm aware that this isn't even the goal of v5, it's a more price-efficient chip, not just a faster chip.

    It was also run a day before the real data, but it shouldn't be ignored b/c there are no other numbers for v5

    You can recognize data from here b/c all of its metrics should be different from the real data by a multiple of 2
    """
    dfnew: pd.DataFrame = dfbase.copy()
    cfg_name: str = "llama2-70b-v5"
    dfnew["run_uid"] = str(uuid4())
    dfnew["config_name"] = cfg_name
    for column in METRICS_TO_FABRICATE:
        dfnew[column] = dfbase[column] * 2

    dfnew["wall_time_utc"] = dfbase["wall_time_utc"] - timedelta(days=1)

    cfgnew = PreTrainingConfig(
        **{
            **asdict(cfgbase),
            "config_name": cfg_name,
            "chip_name": "v5e",
            "flops_per_chip": cfgbase.flops_per_chip * 2,
            "chip_hourly_price": cfgbase.chip_hourly_price / 2,
        }
    )

    return dfnew, cfgnew


def data_for_smol_model(
    dfbase: pd.DataFrame, cfgbase: PreTrainingConfig
) -> tuple[pd.DataFrame, PreTrainingConfig]:
    """
    Same as the base data, except that it goes through 10x the tokens/samples at 1/10 the flops.

    (yes, obviously, flops wouldn't necessarily go down as you'd be running the model more)
    """
    cfg_name: str = cfgbase.config_name + " but smol"
    dfnew: pd.DataFrame = dfbase.copy()
    dfnew["run_uid"] = str(uuid4())
    dfnew["config_name"] = cfg_name
    for column in ["n_tokens", "n_samples"]:
        dfnew[column] = dfbase[column] * 10

    dfnew["model_flops"] = dfbase["model_flops"] / 10

    cfgnew = PreTrainingConfig(
        **{
            **asdict(cfgbase),
            "config_name": cfg_name,
            "n_params": cfgbase.n_params / 10,
        }
    )

    return dfnew, cfgnew


if __name__ == "__main__":
    real_data: pd.DataFrame = pd.read_csv("llama2-70b-test-metrics.csv")
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
    real_data = _format_litgpt_df(real_data, test_training_config, datetime.now())

    data_to_upload: list[tuple[pd.DataFrame, PreTrainingConfig]] = [
        (real_data, test_training_config)
    ]

    for fabrication_fn in [
        old_data_that_should_be_ignored,
        data_for_smol_model,
        data_on_v5,
    ]:
        data_to_upload.append(fabrication_fn(real_data, test_training_config))

    for metrics, metadata in data_to_upload:
        _sync_metadata(metadata)
        _upload_df(metrics, metrics_table)
