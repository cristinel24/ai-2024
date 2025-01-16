import pandas as pd
from sklearn.preprocessing import LabelEncoder


def transform_non_numeric(df: pd.DataFrame):
    label_encoders = {}

    for col in ["Sexe", "Age", "Race", "Abondance", "Logement", "Zone", "Nombre", "Color", "Pattern"]:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le

    return df, label_encoders
