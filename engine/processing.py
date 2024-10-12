import pandas as pd


def _read_dataset_to_df(dataset_csv_path: str) -> pd.DataFrame:
    return pd.read_csv(dataset_csv_path).drop(columns=["Row.names", "Plus"])


def _process_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    missing_values_df = df[df.isnull().any(axis=1)]

    if len(missing_values_df) == 0:
        return df

    print("Missing values!")
    print(missing_values_df.to_string())

    return df.dropna()


def _process_duplicated_values(df: pd.DataFrame) -> pd.DataFrame:
    duplicated_df = df[df.duplicated()]

    if (len(duplicated_df)) == 0:
        return df

    print("Duplicated values!")
    print(duplicated_df.to_string())

    return df.drop_duplicates()


def process_dataset(dataset_csv_path: str) -> pd.DataFrame:
    df = _read_dataset_to_df(dataset_csv_path)
    df = _process_missing_values(df)
    df = _process_duplicated_values(df)

    return df
