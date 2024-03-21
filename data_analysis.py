import pandas as pd

df = pd.read_csv("llama2-70b-test-metrics.csv")
result = df["throughput/batches_per_sec"] / df["throughput/device/batches_per_sec"]


def diff(col: pd.Series) -> pd.Series:
    return (col.iloc[1:] - col.iloc[:-1]).iloc[1:-1]


def is_monotonically_increasing(col: pd.Series) -> bool:
    return (diff(col) > 0).all()


def is_monotonically_nondecreasing(col: pd.Series) -> bool:
    return (diff(col) >= 0).all()


results = pd.DataFrame.from_records(
    [
        {
            "name": col,
            "is_monotonically_increasing": is_monotonically_increasing(df[col]),
            "is_monotonically_nondecreasing": is_monotonically_nondecreasing(df[col]),
        }
        for col in df.columns
    ]
)

print(results)
