import pandas as pd
import matplotlib.pyplot as plt


def instances_per_class(df: pd.DataFrame) -> pd.DataFrame:
    return df["Race"].value_counts()


def df_value_frequency(df: pd.DataFrame) -> list[tuple[str, dict]]:
    return [(attribute, df[attribute].value_counts().to_dict()) for attribute in df.columns]


def plot_value_frequency(df: pd.DataFrame, attribute: str, show=False) -> None:
    plt.close()
    plt.hist(df[attribute], bins=50)
    plt.xlabel(f'{attribute} classes')
    plt.ylabel('Frequency')
    plt.title(f'Value frequency in {attribute}')
    plt.savefig(f"./plots/{attribute}.png")
    if show:
        plt.show()


def plot_attributes_frequencies(df: pd.DataFrame, show: bool) -> None:
    for attribute in df.columns:
        if attribute == 'Horodateur':
            continue
        plot_value_frequency(df, attribute, show=show)


def behavioral_stats(df: pd.DataFrame):
    plt.close()
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)
    ax.set_frame_on(False)

    stats = ['Brutal', 'Dominant', 'Agressif', 'Impulsif', 'Prévisible', 'Distrait']
    table_data = df[stats].describe()

    table = ax.table(cellText=table_data.values, colLabels=table_data.columns, cellLoc='center', loc='center')

    table.scale(1, 2)
    plt.title('Descriptive Statistics of Behavioral Traits', fontsize=16)
    plt.tight_layout()

    plt.savefig("plots/_behavioral_stats.png")
