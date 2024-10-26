from engine.constants import DATASET_PATH
from engine.processing import process_dataset
from engine.statistics import instances_per_class, df_value_frequency, plot_attributes_frequencies, behavioral_stats
from engine.plots import df_correlation
from engine.utils import transform_non_numeric
from pprint import pprint
import matplotlib.pyplot as plt

if __name__ == "__main__":
    df = process_dataset(DATASET_PATH)

    print(instances_per_class(df))
    print(df.to_string())

    df.boxplot(by="Sexe", column=["Calme", "Timide", "Affectueux"], grid=False)
    plt.savefig("plots/boxplot_sexe_calme_timide_affectueux.png")
    df.boxplot(by="Sexe", column=["Solitaire", "Brutal", "Dominant"], grid=False)
    plt.savefig("plots/boxplot_sexe_solitaire_brutal_dominant.png")
    plt.show()

    value_frequencies = df_value_frequency(df)
    for attribute_name, attribute_frequencies in value_frequencies:
        print(f"Number of values for {attribute_name}: {len(attribute_frequencies)}")
        pprint(attribute_frequencies)

    behavioral_stats(df)

    plot_attributes_frequencies(df, show=False)

    transform_non_numeric(df)
    df_correlation(df)
