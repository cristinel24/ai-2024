import pandas as pd
import matplotlib.pyplot as plt


def instances_per_class(df: pd.DataFrame) -> pd.DataFrame:
    return df["Race"].value_counts()

def df_value_frequency(df: pd.DataFrame) -> list[(str, dict)]:
    return [(attribute, df[attribute].value_counts().to_dict()) for attribute in df.columns]

def plot_value_frequency(df: pd.DataFrame, attribute: str) -> None:
    plt.hist(df[attribute], bins=50)
    plt.xlabel(f'{attribute} classes')
    plt.ylabel('Frequency')
    plt.title(f'Value frequency in {attribute}')
    plt.show()

def plot_attributes_frequencies(df: pd.DataFrame) -> None:
    for attribute in df.columns:
        plot_value_frequency(df, attribute)