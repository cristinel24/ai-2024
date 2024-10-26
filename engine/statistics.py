import pandas as pd


def compute_instances_per_class_df(df: pd.DataFrame) -> pd.DataFrame:
    return df["Race"].value_counts()


def compute_value_frequency_stats(df: pd.DataFrame) -> list[tuple[str, dict]]:
    return [(attribute, df[attribute].value_counts().to_dict()) for attribute in df.columns
            if attribute != "Horodateur"]
