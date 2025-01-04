import pandas as pd
import matplotlib.pyplot as plt


def df_correlation(df: pd.DataFrame):
    plt.close()
    columns = df.columns
    correlation_matrix = df[columns].corr()

    plt.figure(figsize=(12, 10))
    plt.matshow(correlation_matrix, fignum=1)
    plt.colorbar()
    plt.xticks(range(len(columns)), columns, rotation=90)
    plt.yticks(range(len(columns)), columns)
    plt.title("Attribute correlation")
    plt.show()


