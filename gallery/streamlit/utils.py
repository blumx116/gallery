from math import floor, log
from typing import Any

from gallery.streamlit.queries import Primitive


def human_readable_bignum(num: float) -> str:
    """
    Prints numbers like 70000000000 -> 70B
    """
    magnitude: int = int(floor(log(num, 1000)))
    suffix: str = ["", "K", "M", "B", "T"][magnitude]
    num /= 1000**magnitude
    formatted_num: str = f"{num:.2f}".strip("0").strip(".")
    return f"{formatted_num}{suffix}"


def value_formatter(value: Primitive) -> Primitive:
    if isinstance(value, float):
        return human_readable_bignum(value)
    return value


def dict_diff(dict1: dict[str, Any], dict2: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in dict1.items()
        if key not in dict2 or dict2[key] != value
    }


def pretty_dict(d: dict[str, Any]) -> str:
    result: str = ""
    for key, value in d.items():
        result += f"{key}: {value_formatter(value)}\n"
    return result
