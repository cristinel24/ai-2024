import sys
import os
import random
import argparse
import re
from collections import defaultdict, Counter
import joblib
from rake_nltk import Rake
import nltk
from langdetect import detect
from nltk.corpus import wordnet as wn
from nltk.tokenize import word_tokenize
from word2number import w2n
from googletrans import Translator
import openai


def load_label_encoders(encoders_path: str) -> dict:
    return joblib.load(encoders_path)


def read_text(input_path: str = None) -> str:
    if input_path and os.path.isfile(input_path):
        with open(input_path, 'r', encoding='utf-8') as f:
            text = f.read()
    else:
        print("Enter text:")
        text = sys.stdin.read()
    return text.strip()


def detect_language(text: str) -> str:
    try:
        language_code = detect(text)
    except:
        language_code = "unknown"
    return language_code


def translate_to_english(text: str) -> str:
    translator = Translator()
    lang = detect(text)
    print(f"LANG Text: {lang}")
    result = translator.translate(text, src=lang, dest='en')
    return result.text


def get_stylometry_info(text: str):
    words = word_tokenize(text)
    word_count = len(words)
    char_count = len(text)
    freqs = Counter(words)
    return {
        'word_count': word_count,
        'char_count': char_count,
        'freqs': freqs
    }


def find_word_variants(word: str):
    """
    synonyms, hypernyms using wordnet
    """
    variants = set()
    if not word.isalpha():
        return []
    synsets = wn.synsets(word, lang='eng')
    for syn in synsets:
        for lemma in syn.lemmas():
            variants.add(lemma.name().replace("_", " "))
        for hyper in syn.hypernyms():
            for lemma in hyper.lemmas():
                variants.add(lemma.name().replace("_", " "))
        for lemma in syn.lemmas():
            if lemma.antonyms():
                for ant_lemma in lemma.antonyms():
                    variants.add(f"not {ant_lemma.name().replace('_', ' ')}")
    if word in variants:
        variants.remove(word)
    return list(variants)


def replace_words_with_variants(text: str, ratio=0.2) -> str:
    words = word_tokenize(text)
    if not words:
        return text
    num_to_replace = int(len(words) * ratio)
    if num_to_replace <= 0:
        return text

    indices_to_replace = random.sample(range(len(words)), num_to_replace)
    new_words = words[:]
    for idx in indices_to_replace:
        w = words[idx].lower()
        variants = find_word_variants(w)
        if variants:
            chosen = random.choice(variants)
            if words[idx][0].isupper():
                chosen = chosen.capitalize()
            new_words[idx] = chosen
    return " ".join(new_words)


def extract_keywords(text: str, top_n=5):
    words = [w.lower() for w in word_tokenize(text) if w.isalpha()]
    freqs = Counter(words)
    return [w for w, _ in freqs.most_common(top_n)]


def generate_sentences_for_keywords(original_text, top_n=5):
    rake_nltk_var = Rake()

    rake_nltk_var.extract_keywords_from_text(original_text)
    top_keywords = rake_nltk_var.get_ranked_phrases()[:top_n]

    print("\nTop Keywords:")
    print(top_keywords)

    sentences = []
    for kw in top_keywords:
        messages = [
            {
                "role": "system",
                "content": "You are a helpful assistant."
            },
            {
                "role": "user",
                "content": f"Generate one meaningful sentence that includes the phrase in ROMANIAN LANGUAGE: '{kw}'.\nSentence:"
            }
        ]

        response = openai.ChatCompletion.create(
            model="gpt-4o-mini",
            messages=messages,
            max_tokens=50,
            temperature=0.7,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0
        )

        sentence = response.choices[0].message["content"].strip()
        sentences.append(sentence)
    return sentences


def parse_english_sentence_to_cat_attributes(text: str,
                                             color_encoder=None,
                                             pattern_encoder=None,
                                             zone_encoder=None) -> dict:
    cat_dict = {
        'Sexe': 0,
        'Age': 0,
        'Nombre': 0,
        'Logement': 0,
        'Zone': 0,
        'Color': 0,
        'Pattern': 0,

        'Ext': 0,
        'Obs': 0,
        'Timide': 0,
        'Calme': 0,
        'Effraye': 0,
        'Intelligent': 0,
        'Vigilant': 0,
        'Perseverant': 0,
        'Affectueux': 0,
        'Amical': 0,
        'Solitaire': 0,
        'Brutal': 0,
        'Dominant': 0,
        'Agressif': 0,
        'Impulsif': 0,
        'Previsible': 0,
        'Distrait': 0,

        'Abondance': 0,
        'PredOiseau': 0,
        'PredMamm': 0
    }

    attribute_synonyms = {
        'Ext': {"extroverted", "outgoing", "sociable"},
        'Obs': {"observant", "attentive", "watchful"},
        'Timide': {"timid", "shy", "introverted", "bashful", "reserved"},
        'Calme': {"calm", "peaceful", "tranquil", "serene", "relaxed"},
        'Effraye': {"afraid", "scared", "fearful", "frightened", "terrified"},
        'Intelligent': {"intelligent", "smart", "clever", "bright", "brainy"},
        'Vigilant': {"vigilant", "alert", "aware", "cautious"},
        'Perseverant': {"perseverant", "persistent", "determined", "tenacious"},
        'Affectueux': {"affectionate", "loving", "friendly", "cuddly", "warm"},
        'Amical': {"amiable", "amical", "friendly", "kind", "cordial"},
        'Solitaire': {"solitary", "loner", "independent"},
        'Brutal': {"brutal", "violent", "vicious", "savage"},
        'Dominant': {"dominant", "bossy", "territorial", "assertive", "leader"},
        'Agressif': {"aggressive", "hostile", "antagonistic", "belligerent"},
        'Impulsif': {"impulsive", "rash", "hasty", "spontaneous"},
        'Previsible': {"predictable", "foreseeable", "expected"},
        'Distrait': {"distracted", "inattentive", "spacey", "absentminded"},
    }

    intensifiers = {
        "slightly": -1,
        "somewhat": -1,
        "fairly": -1,
        "mildly": -1,
        "a bit": -1,

        "very": +3,
        "quite": +4,
        "extremely": +5,
        "incredibly": +6,
        "remarkably": +7,
        "truly": +8,
        "highly": +9,
        "super": +9,
        "insanely": +10,
    }

    sex_synonyms = {
        "male": 1, "boy": 1, "masculine": 1,
        "female": 0, "girl": 0, "feminine": 0
    }

    abundance_synonyms = {
        "scarce": 1, "limited": 1,
        "moderate": 2, "enough": 2,
        "plentiful": 3, "abundant": 3, "lots": 3
    }
    bird_synonyms = {"bird", "birds", "sparrow", "pigeon", "parrot"}
    mamm_synonyms = {"mouse", "mice", "rat", "squirrel", "hamster"}

    tokens = nltk.word_tokenize(text)
    lower_tokens = [t.lower() for t in tokens]
    lower_text = text.lower()

    for tok in lower_tokens:
        if tok in sex_synonyms:
            cat_dict['Sexe'] = sex_synonyms[tok]
            break

    if color_encoder is not None:
        color_classes = [c.lower() for c in color_encoder.classes_]
        found_colors = [tok for tok in lower_tokens if tok in color_classes]
        if found_colors:
            chosen_color = found_colors[0]
            try:
                cat_dict['Color'] = int(color_encoder.transform([chosen_color])[0])
            except ValueError:
                cat_dict['Color'] = 0

    if pattern_encoder is not None:
        pattern_classes = [c.lower() for c in pattern_encoder.classes_]
        found_patterns = [tok for tok in lower_tokens if tok in pattern_classes]
        if found_patterns:
            chosen_pattern = found_patterns[0]
            try:
                cat_dict['Pattern'] = int(pattern_encoder.transform([chosen_pattern])[0])
            except ValueError:
                cat_dict['Pattern'] = 0

    if zone_encoder is not None:
        zone_classes = [c.lower() for c in zone_encoder.classes_]
        found_zones = [tok for tok in lower_tokens if tok in zone_classes]
        if found_zones:
            chosen_pattern = found_zones[0]
            try:
                cat_dict['Zone'] = int(zone_encoder.transform([chosen_pattern])[0])
            except ValueError:
                cat_dict['Zone'] = 0

    match_age_num = re.findall(r"(\d+(?:\.\d+)?)\s*years?\s*old", lower_text)
    if match_age_num:
        cat_dict['Age'] = int(float(match_age_num[0]))

    match_age_hyphen = re.findall(r"(\d+(?:\.\d+)?)\s*-\s*year\s*-\s*old", lower_text)
    if match_age_hyphen:
        cat_dict['Age'] = int(float(match_age_hyphen[0]))

    match_age_text = re.findall(r"([a-zA-Z]+)\s+years?\s+old", lower_text)
    if match_age_text:
        try:
            cat_dict['Age'] = w2n.word_to_num(match_age_text[0])
        except:
            cat_dict['Age'] = 0

    attribute_offsets = defaultdict(int)
    current_intensity = 0

    for i, token in enumerate(lower_tokens):
        if token in intensifiers:
            current_intensity += intensifiers[token]
            continue
        for attr, synonyms in attribute_synonyms.items():
            if token in synonyms:
                attribute_offsets[attr] += current_intensity
                current_intensity = 0
                break

    for attr in attribute_synonyms.keys():
        base_val = cat_dict[attr]
        offset = attribute_offsets[attr]
        new_val = max(1, min(9, base_val + offset))
        cat_dict[attr] = new_val

    if re.search(r"\blives?\s+in\s+an?\s+apartment\b", lower_text):
        cat_dict['Logement'] = 0
    elif re.search(r"\blives?\s+in\s+a?\s+house\b", lower_text):
        cat_dict['Logement'] = 2
    elif re.search(r"\blives?\s+outdoors?\b", lower_text):
        cat_dict['Logement'] = 3


    match_nombre_num = re.findall(r"(\d+)\s+(?:cats|companions|kittens|felines)", lower_text)
    if match_nombre_num:
        cat_dict['Nombre'] = int(match_nombre_num[0])

    match_nombre_text = re.findall(r"([a-zA-Z]+)\s+(?:cats|companions|kittens|felines)", lower_text)
    if match_nombre_text:
        try:
            cat_dict['Nombre'] = w2n.word_to_num(match_nombre_text[0])
        except:
            cat_dict['Nombre'] = 0

    for tok in lower_tokens:
        if tok in abundance_synonyms:
            cat_dict['Abondance'] = abundance_synonyms[tok]

    for tok in lower_tokens:
        if tok in bird_synonyms:
            cat_dict['PredOiseau'] = 1
        if tok in mamm_synonyms:
            cat_dict['PredMamm'] = 1

    if re.search(r"hunts?\s+birds?", lower_text):
        cat_dict['PredOiseau'] = 1
    if re.search(r"hunts?\s+mice|rats?", lower_text):
        cat_dict['PredMamm'] = 1

    antonym_pairs = [
        ("Ext", "Timide"),
        ("Calme", "Agressif"),
        ("Solitaire", "Amical"),
        ("Obs", "Distrait"),
        ("Effraye", "Dominant"),
        ("Brutal", "Affectueux"),
        ("Perseverant", "Impulsif")
    ]

    for a1, a2 in antonym_pairs:
        val1 = cat_dict[a1]
        val2 = cat_dict[a2]
        if val1 > 2:
            diff = val1 - 2
            cat_dict[a2] = max(1, val2 - diff)
        val2 = cat_dict[a2]
        if val2 > 2:
            diff2 = val2 - 2
            cat_dict[a1] = max(1, cat_dict[a1] - diff2)

    return cat_dict


def describe_race(race: str):
    messages = [
        {
            "role": "system",
            "content": "You are a helpful assistant."
        },
        {
            "role": "user",
            "content": f"Generate one meaningful description of the cat race: {race}. "
                       f"This is the legend: BEN/SBI/BRI/CHA/EUR/MCO/PER/RAG/SPH/ORI/TUV/ Autre/NSP = Bengal/ Birman/ British Shorthair/ Chartreux / European/ Maine coon / Persian/ Ragdoll/ Savannah / Sphynx/ Siamese/ Turkish angora / No breed / Other / Unknown "
                       f"The text you generate MUST be in romanian. Return only the description and nothing else more"
        }
    ]

    response = openai.ChatCompletion.create(
        model="gpt-4o-mini",
        messages=messages,
        temperature=0.7,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0
    )

    sentences = []
    for sentence in response.choices:
        sentences.append(sentence.message["content"].strip())
    return " ".join(sentences)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test Romanian -> English NLP pipeline for cat attributes.")
    parser.add_argument("--textfile", help="Path to input Romanian text file", default=None)
    parser.add_argument("--encoders", help="Path to label_encoders.pkl", default="../label_encoders.pkl")

    args = parser.parse_args()

    try:
        label_encoders = load_label_encoders(args.encoders)
        color_encoder = label_encoders["Color"]
        pattern_encoder = label_encoders["Pattern"]
        zone_encoder = label_encoders["Zone"]
    except:
        color_encoder = None
        pattern_encoder = None
        zone_encoder = None

    romanian_text = read_text(args.textfile)
    print("\n[Original Romanian text]")
    print(romanian_text)

    english_text = translate_to_english(romanian_text)
    print("\n[Translated English text]")
    print(english_text)

    info = get_stylometry_info(english_text)
    print("\n[Stylometry Info - English]")
    print(f"Word count: {info['word_count']}")
    print(f"Char count: {info['char_count']}")
    print(f"Most common words: {info['freqs'].most_common(5)}")

    alt_text = replace_words_with_variants(english_text, ratio=0.2)
    print("\n[Alternative English text - 20% replaced]")
    print(alt_text)

    keywords = extract_keywords(english_text, top_n=5)
    print("\n[Top Keywords in English]")
    print(keywords)
    key_sentences = generate_sentences_for_keywords(keywords)
    print("\n[Sentences for each keyword]")
    for s in key_sentences:
        print(s)

    cat_attributes = parse_english_sentence_to_cat_attributes(
        english_text,
        color_encoder=color_encoder,
        pattern_encoder=pattern_encoder,
        zone_encoder=zone_encoder
    )
    print("\n[Cat Attributes from English text]")
    print(cat_attributes)
