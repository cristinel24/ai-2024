from engine.constants import DATASET_PATH
from engine.plots import (plot_boxplot_calm_timid_affective_df, plot_boxplot_loner_brutal_dominant_df,
                          plot_behavioral_stats_df, plot_attributes_frequencies_dfs, plot_columns_correlation_df)
from engine.processing import process_dataset
from pprint import pprint
from engine.statistics import compute_instances_per_class_df, compute_value_frequency_stats
from engine.utils import transform_non_numeric_attributes

if __name__ == "__main__":
    df = process_dataset(DATASET_PATH)

    print(f"Instances per class:\n{compute_instances_per_class_df(df).to_string()}")
    print(f"Processed dataset:\n{df}")

    plot_boxplot_calm_timid_affective_df(df)
    plot_boxplot_loner_brutal_dominant_df(df)

    value_frequency_stats = compute_value_frequency_stats(df)
    for attribute_name, attribute_frequencies in value_frequency_stats:
        print(f"Number of values for {attribute_name}: {len(attribute_frequencies)}")
        pprint(attribute_frequencies)

    plot_behavioral_stats_df(df)
    plot_attributes_frequencies_dfs(df)

    transform_non_numeric_attributes(df)
    plot_columns_correlation_df(df)
