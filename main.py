from engine.constants import DATASET_PATH
from engine.processing import process_dataset
from engine.statistics import instances_per_class, df_value_frequency, plot_attributes_frequencies, behavioral_stats
from engine.plots import df_correlation
from engine.utils import transform_non_numeric
from pprint import pprint
from mlp.model import MLPModel
import matplotlib.pyplot as plt


if __name__ == "__main__":
    df = process_dataset(DATASET_PATH, use_smote=True)
    transform_non_numeric(df)

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

    df_correlation(df)

    model = MLPModel(df, 100, 0.1, 200)
    try:
        model.load_model()
    except Exception as e:
        print(f"No pre-trained model! Error: {e}")
        print("Training...")
        model.train(50)

    print(f"Best accuracy: {model.best_accuracy * 100:.2f} ")
