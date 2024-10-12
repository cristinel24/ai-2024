import pandas as pd


def instances_per_class(df: pd.DataFrame) -> pd.DataFrame:
    return df["Race"].value_counts()
