import pandas as pd
from sklearn.preprocessing import LabelEncoder


def transform_non_numeric(df: pd.DataFrame):
    df['Sexe'] = LabelEncoder().fit_transform(df['Sexe'])
    df['Age'] = LabelEncoder().fit_transform(df['Age'])
    df['Race'] = LabelEncoder().fit_transform(df['Race'])
    df['Abondance'] = LabelEncoder().fit_transform(df['Abondance'])
    df['Logement'] = LabelEncoder().fit_transform(df['Logement'])
    df['Zone'] = LabelEncoder().fit_transform(df['Zone'])
    df['Nombre'] = LabelEncoder().fit_transform(df['Nombre'])

