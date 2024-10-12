from engine.constants import DATASET_PATH
from engine.processing import process_dataset
from engine.statistics import instances_per_class, df_value_frequency, plot_value_frequency, plot_attributes_frequencies

if __name__ == "__main__":
    df = process_dataset(DATASET_PATH)

    print(instances_per_class(df))
    print(df.to_string())

    value_frequencies = df_value_frequency(df)
    for attribute_name, attribute_frequencies in value_frequencies:
        print(f"Number of values for {attribute_name}: {len(attribute_frequencies)}")
        print(attribute_frequencies)

    plot_attributes_frequencies(df)