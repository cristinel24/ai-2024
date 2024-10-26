import random
import numpy as np
import pandas as pd
from engine.mappings import BREED_TO_COLOR_PATTERN


def _read_dataset_to_df(dataset_csv_path: str) -> pd.DataFrame:
    return pd.read_csv(dataset_csv_path).drop(columns=["Row.names", "Plus"])


def _impute_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    for column in df.select_dtypes(include=['float64', 'int64']).columns:
        median_value = df[column].median()
        df[column] = df[column].fillna(median_value)

    for column in df.select_dtypes(include=['object']).columns:
        mode_value = df[column].mode()[0]
        df[column] = df[column].fillna(mode_value)

    return df


def _process_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    df['Abondance'] = df['Abondance'].replace("NSP", np.nan)
    missing_values_df = df[df.isnull().any(axis=1)]

    if len(missing_values_df) == 0:
        return df

    print("Missing values!")
    print(missing_values_df)

    return _impute_missing_values(df)


def _process_duplicated_values(df: pd.DataFrame) -> pd.DataFrame:
    duplicated_df = df[df.duplicated()]

    if len(duplicated_df) == 0:
        return df

    print("Duplicated values!")
    print(duplicated_df)

    return df.drop_duplicates()


def _get_random_attribute(breed: str, attribute_type: str) -> list[str]:
    assert attribute_type in ['color', 'pattern']

    if attribute_type == 'color':
        return random.choice(BREED_TO_COLOR_PATTERN.get(breed, ('Various', 'Various'))[0].split(', '))
    elif attribute_type == 'pattern':
        return random.choice(BREED_TO_COLOR_PATTERN.get(breed, ('Various', 'Various'))[1].split(', '))


def _add_new_attributes(df: pd.DataFrame) -> pd.DataFrame:
    df['Color'] = df['Race'].map(lambda breed: _get_random_attribute(breed, 'color'))
    df['Pattern'] = df['Race'].map(lambda breed: _get_random_attribute(breed, 'pattern'))

    return df


def process_dataset(dataset_csv_path: str) -> pd.DataFrame:
    df = _read_dataset_to_df(dataset_csv_path)
    df = _process_missing_values(df)
    df = _add_new_attributes(df)
    df = _process_duplicated_values(df)

    return df
