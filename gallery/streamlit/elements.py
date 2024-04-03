from typing import Any, Callable, Optional

import streamlit as st

from gallery.constants import metadata_table
from gallery.streamlit.queries import Primitive, query_unique
from gallery.streamlit.utils import human_readable_bignum, value_formatter


def _r_filter_config(
    named_params: list[tuple[str, str]],
    constraints: dict[str, Primitive],
    widget_key_prefix: str,
) -> dict[str, Primitive]:
    """
    This function is written in a recursive form b/c it works naturally with the fact that we don't want to modify
    the `constraints` dictionary in place.

    This is because caching by streamlit is done by checking if any arguments change, and this would result in a lot of
    re-rendering any time we updated the constraints dict.

    Instead, we simply copy the dict upon each recursive function call, which is inefficient, but trivial.
    """

    if len(named_params) == 0:
        return constraints

    (db_name, display_name), rest = named_params[0], named_params[1:]

    possible_values: list[Primitive] = query_unique(
        metadata_table, target=db_name, **constraints
    )

    filtered_value: Optional[Primitive] = st.radio(
        display_name,
        possible_values,
        key=widget_key_prefix + "_radio_" + db_name,
        format_func=value_formatter,
    )

    forwarded_constraints: dict[str, Primitive] = {
        db_name: filtered_value,
        **constraints,
    }

    return _r_filter_config(rest, forwarded_constraints, widget_key_prefix)


def filter_config(
    named_params: list[tuple[str, str]], widget_key_prefix: Optional[str] = None
) -> dict[str, Primitive]:
    """
    Generates a set of radio buttons for the user to filter values for.
    For each param, it will generate a radio button allowing the user to select a value from the database.
    Subsequent calls to the database may then be filtered by that value of the param.

    Furthermore, each radio button generated is aware of the selections for each of the previous radio buttons
    So it will only show options for which there is at least one configuration possible that coheres with the previous selections.
    Args:
        named_params: [(db_name, display_name)]
            db_name: str is the name of the column in the database to be filtered by
            display_name: str is the name that will be displayed alongside the corresponding radio button
        widget_key: optional argument used for giving radio buttons unique keys
            this isn't necessary if you're calling filter_config with unique database columns every time
            but is necessary to avoid naming conflicts if multiple configs use the same columns

    Returns:
        constraints: the values selected by the user.
            any radio buttons where the user did not choose to filter are omitted
            from this dictionary

            example: {
                "ml_framework": "jax",
                "max_seq_len": 2048
            }
    """

    if widget_key_prefix is None:
        widget_key_prefix = ""
    return _r_filter_config(named_params, {}, widget_key_prefix)
