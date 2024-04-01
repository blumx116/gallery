from math import floor, log


def human_readable_bignum(num: float) -> str:
    """
    Prints numbers like 70000000000 -> 70B
    """
    magnitude: int = int(floor(log(num, 1000)))
    suffix: str = ["", "K", "M", "B", "T"][magnitude]
    num /= 1000**magnitude
    formatted_num: str = f"{num:.2f}".strip("0").strip(".")
    return f"{formatted_num}{suffix}"
