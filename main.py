from engine.constants import DATASET_PATH
from engine.processing import process_dataset
from engine.statistics import instances_per_class, df_value_frequency, plot_attributes_frequencies, behavioral_stats
from engine.plots import df_correlation
from engine.utils import transform_non_numeric
from pprint import pprint
from mlp.model import MLPModel
import matplotlib.pyplot as plt
from engine.text_processing import read_text, translate_to_english, parse_english_sentence_to_cat_attributes, replace_words_with_variants, get_stylometry_info, extract_keywords, generate_sentences_for_keywords
import numpy as np


if __name__ == "__main__":
    df = process_dataset(DATASET_PATH, use_smote=True)
    df, label_encoders = transform_non_numeric(df)

    # import joblib
    # joblib.dump(label_encoders, "label_encoders.pkl")

    # print(instances_per_class(df))
    #
    # print(df.to_string())

    # df.boxplot(by="Sexe", column=["Calme", "Timide", "Affectueux"], grid=False)
    # plt.savefig("plots/boxplot_sexe_calme_timide_affectueux.png")
    # df.boxplot(by="Sexe", column=["Solitaire", "Brutal", "Dominant"], grid=False)
    # plt.savefig("plots/boxplot_sexe_solitaire_brutal_dominant.png")
    # plt.show()
    #
    # value_frequencies = df_value_frequency(df)
    # for attribute_name, attribute_frequencies in value_frequencies:
    #     print(f"Number of values for {attribute_name}: {len(attribute_frequencies)}")
    #     pprint(attribute_frequencies)

    # behavioral_stats(df)
    #
    # plot_attributes_frequencies(df, show=False)

    # df_correlation(df)

    model = MLPModel(df, 100, 0.1, 200)
    try:
        model.load_model()
    except Exception as e:
        print(f"No pre-trained model! Error: {e}")
        print("Training...")
        model.train(50)

    print(f"Best accuracy: {model.best_accuracy * 100:.2f} ")

    print("\nEnter text:")
    romanian_text = read_text(input_path=None)

    english_text = translate_to_english(romanian_text)
    print("\nTranslated:")
    print(english_text)

    info = get_stylometry_info(english_text)
    print("\nStylometry Info")
    print(f"Word count: {info['word_count']}")
    print(f"Char count: {info['char_count']}")
    print(f"Most common words: {info['freqs'].most_common(5)}")

    alt_text = replace_words_with_variants(english_text, ratio=0.2)
    print("\nAlternative text - 20% replaced:")
    print(alt_text)

    key_sentences = generate_sentences_for_keywords(english_text)
    print("\nSentences for each keyword:")
    for s in key_sentences:
        print(s)

    color_encoder = label_encoders["Color"]
    pattern_encoder = label_encoders["Pattern"]
    zone_encoder = label_encoders["Zone"]

    cat_attributes = parse_english_sentence_to_cat_attributes(
        english_text,
        color_encoder=color_encoder,
        pattern_encoder=pattern_encoder,
        zone_encoder=zone_encoder
    )
    print("\n[Parsed cat attributes]")
    pprint(cat_attributes)

    test_vector = [
        cat_attributes['Sexe'],
        cat_attributes['Age'],
        cat_attributes['Nombre'],
        cat_attributes['Logement'],
        cat_attributes['Zone'],
        cat_attributes['Ext'],
        cat_attributes['Obs'],
        cat_attributes['Timide'],
        cat_attributes['Calme'],
        cat_attributes['Effraye'],
        cat_attributes['Intelligent'],
        cat_attributes['Vigilant'],
        cat_attributes['Perseverant'],
        cat_attributes['Affectueux'],
        cat_attributes['Amical'],
        cat_attributes['Solitaire'],
        cat_attributes['Brutal'],
        cat_attributes['Dominant'],
        cat_attributes['Agressif'],
        cat_attributes['Impulsif'],
        cat_attributes['Previsible'],
        cat_attributes['Distrait'],
        cat_attributes['Abondance'],
        cat_attributes['PredOiseau'],
        cat_attributes['PredMamm'],
        cat_attributes['Color'],
        cat_attributes['Pattern']
    ]
    test_vector = np.array(test_vector).reshape(1, -1)

    _, _, predictions = model._forward_propagation(test_vector)
    predicted_race_idx = np.argmax(predictions, axis=1)[0]

    race_encoder = label_encoders["Race"]
    predicted_race_label = race_encoder.inverse_transform([predicted_race_idx])[0]

    print(f"\nPREDICTED RACE for this cat: {predicted_race_label}")
