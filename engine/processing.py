import random
import numpy as np
import pandas as pd


def _read_dataset_to_df(dataset_csv_path: str) -> pd.DataFrame:
    return pd.read_csv(dataset_csv_path).drop(columns=["Row.names", "Plus"])


def _process_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    df['Abondance'] = df['Abondance'].replace("NSP", np.nan)
    missing_values_df = df[df.isnull().any(axis=1)]

    if len(missing_values_df) == 0:
        return df

    print("Missing values!")
    print(missing_values_df.to_string())

    return df


def _impute_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    for column in df.select_dtypes(include=['float64', 'int64']).columns:
        median_value = df[column].median()
        df[column].fillna(median_value, inplace=True)

    for column in df.select_dtypes(include=['object']).columns:
        mode_value = df[column].mode()[0]
        df[column].fillna(mode_value, inplace=True)

    return df


def _process_duplicated_values(df: pd.DataFrame) -> pd.DataFrame:
    duplicated_df = df[df.duplicated()]

    if (len(duplicated_df)) == 0:
        return df

    print("Duplicated values!")
    print(duplicated_df.to_string())

    return df.drop_duplicates()


def _add_new_attributes(df: pd.DataFrame) -> pd.DataFrame:

    def get_random_attribute(breed, attribute_type):
        if attribute_type == 'color':
            return random.choice(breed_color_pattern_updated.get(breed, ('Various', 'Various'))[0].split(', '))
        elif attribute_type == 'pattern':
            return random.choice(breed_color_pattern_updated.get(breed, ('Various', 'Various'))[1].split(', '))

    breed_color_pattern_updated = {
        'BEN': ('Brown, Silver, Snow', 'Spotted, Marbled'),
        'SBI': ('Cream, Seal, Blue, Chocolate, Lilac', 'Colorpoint'),
        'BRI': ('Blue, Black, White, Cream', 'Solid, Tabby, Bicolor'),
        'CHA': ('Blue-gray', 'Solid'),
        'EUR': ('Tabby, Black, White, Gray, Cream', 'Tabby, Solid, Bicolor'),
        'MCO': ('Brown, Red, Cream, Blue', 'Tabby, Solid, Tortie, Bicolor'),
        'PER': ('White, Black, Blue, Red, Cream, Tortoiseshell', 'Solid, Tabby, Shaded, Smoke, Tortie'),
        'RAG': ('White, Seal, Blue, Chocolate, Lilac', 'Colorpoint, Mitted, Bicolor'),
        'SPH': ('Brown, Silver, Black', 'Spotted, Marbled'),
        'ORI': ('Black, White, Cream, Red, Blue, Tortie', 'Solid, Bicolor, Tortie'),
        'TUV': ('White, Black, Blue, Red, Cream', 'Solid, Tabby, Bicolor'),
        'Autre': ('Various', 'Various'),
        'NSP': ('Various', 'Various')
    }

    df['Color'] = df['Race'].map(lambda breed: get_random_attribute(breed, 'color'))
    df['Pattern'] = df['Race'].map(lambda breed: get_random_attribute(breed, 'pattern'))

    return df


def process_dataset(dataset_csv_path: str) -> pd.DataFrame:
    df = _read_dataset_to_df(dataset_csv_path)
    df = _process_missing_values(df)
    df = _impute_missing_values(df)
    df = _add_new_attributes(df)
    df = _process_duplicated_values(df)

    return df
