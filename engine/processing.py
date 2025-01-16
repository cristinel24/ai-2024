import random
from sklearn.utils import resample
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTENC
from sklearn.preprocessing import LabelEncoder

def _read_dataset_to_df(dataset_csv_path: str) -> pd.DataFrame:
    return pd.read_csv(dataset_csv_path).drop(columns=["Row.names", "Plus", "Horodateur"])


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


def _balance_dataset_smote(df: pd.DataFrame, target_column: str) -> pd.DataFrame:
    x = df.drop(target_column, axis=1)
    y = df[target_column]

    categorical_features = x.select_dtypes(include=['object', 'category']).columns.tolist()
    categorical_indices = [x.columns.get_loc(col) for col in categorical_features]

    for col in categorical_features:
        le = LabelEncoder()
        x[col] = le.fit_transform(x[col])

    if y.dtype == 'object' or str(y.dtype) == 'category':
        y_le = LabelEncoder()
        y = y_le.fit_transform(y)
    else:
        y_le = None

    smote_nc = SMOTENC(categorical_features=categorical_indices, random_state=42)
    x_resampled, y_resampled = smote_nc.fit_resample(x, y)

    for col in categorical_features:
        le = LabelEncoder()
        le.fit(df[col])
        x_resampled[col] = le.inverse_transform(x_resampled[col])

    if y_le is not None:
        y_resampled = y_le.inverse_transform(y_resampled)

    resampled_df = pd.concat([pd.DataFrame(x_resampled, columns=x.columns), pd.Series(y_resampled, name=target_column)], axis=1)

    return resampled_df


def _balance_dataset(df: pd.DataFrame, target_column: str) -> pd.DataFrame:
    majority_class = df[target_column].value_counts().idxmax()
    majority_count = df[target_column].value_counts().max()

    df_majority = df[df[target_column] == majority_class]
    resampled_dfs = [df_majority]

    for class_label in df[target_column].unique():
        if class_label != majority_class:
            df_minority = df[df[target_column] == class_label]
            df_upsampled = resample(
                df_minority,
                replace=True,
                n_samples=majority_count,
                random_state=42
            )
            resampled_dfs.append(df_upsampled)

    balanced_df = pd.concat(resampled_dfs, axis=0).sample(frac=1, random_state=42).reset_index(drop=True)

    return balanced_df


def process_dataset(dataset_csv_path: str, use_smote=False) -> pd.DataFrame:
    df = _read_dataset_to_df(dataset_csv_path)
    df = _process_missing_values(df)
    df = _impute_missing_values(df)
    df = _add_new_attributes(df)
    df = _process_duplicated_values(df)
    if use_smote:
        df = _balance_dataset_smote(df, "Race")
    else:
        df = _balance_dataset(df, "Race")

    string_cols = df.select_dtypes(include='object').columns
    df[string_cols] = df[string_cols].apply(lambda col: col.str.lower())

    return df
