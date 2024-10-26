import os
import pandas as pd
import matplotlib.pyplot as plt
from engine.constants import PLOTS_PATH


def plot_columns_correlation_df(df: pd.DataFrame, show_plot: bool = False) -> None:
    plt.close()

    columns = df.columns.drop('Horodateur')
    columns_correlation_df = df[columns].corr()

    plt.figure(figsize=(12, 10))
    plt.matshow(columns_correlation_df, fignum=1)
    plt.colorbar()
    plt.xticks(range(len(columns)), columns, rotation=90)
    plt.yticks(range(len(columns)), columns)
    plt.title("Attribute correlation")

    plt.savefig(os.path.join(PLOTS_PATH, "columns_correlation.png"))
    if show_plot:
        plt.show()


def _plot_attribute_frequency_df(df: pd.DataFrame, attribute: str, show_plot: bool = False) -> None:
    plt.close()

    plt.hist(df[attribute], bins=50)
    plt.xlabel(f'{attribute} classes')
    plt.ylabel('Frequency')
    plt.title(f'Value frequency in {attribute}')

    plt.savefig(os.path.join(PLOTS_PATH, f"{attribute}_frequency.png"))
    if show_plot:
        plt.show()


def plot_attributes_frequencies_dfs(df: pd.DataFrame, show_plots: bool = False) -> None:
    for attribute in df.columns:
        if attribute == 'Horodateur':
            continue
        _plot_attribute_frequency_df(df, attribute, show_plots)


def plot_behavioral_stats_df(df: pd.DataFrame, show_plot: bool = False) -> None:
    plt.close()

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)
    ax.set_frame_on(False)

    stats = ['Brutal', 'Dominant', 'Agressif', 'Impulsif', 'Prévisible', 'Distrait']
    table_data = df[stats].describe().reset_index()

    table = ax.table(cellText=table_data.values, colLabels=table_data.columns, cellLoc='center', loc='center')

    table.scale(1, 2)
    plt.title('Descriptive Statistics of Behavioral Traits', fontsize=16)
    plt.tight_layout()

    plt.savefig(os.path.join(PLOTS_PATH, "behavioral_stats.png"))
    if show_plot:
        plt.show()


def plot_boxplot_calm_timid_affective_df(df: pd.DataFrame, show_plot: bool = False) -> None:
    plt.close()

    df.boxplot(by="Sexe", column=["Calme", "Timide", "Affectueux"], grid=False)

    plt.savefig(os.path.join(PLOTS_PATH, "boxplot_calm_timid_affective.png"))
    if show_plot:
        plt.show()

def plot_boxplot_loner_brutal_dominant_df(df: pd.DataFrame, show_plot: bool = False) -> None:
    plt.close()

    df.boxplot(by="Sexe", column=["Solitaire", "Brutal", "Dominant"], grid=False)

    plt.savefig(os.path.join(PLOTS_PATH, "boxplot_loner_brutal_dominant.png"))
    if show_plot:
        plt.show()
