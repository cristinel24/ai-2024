from engine.constants import DATASET_PATH
from engine.processing import process_dataset
from engine.statistics import instances_per_class
from engine.plots import df_correlation
from engine.utils import transform_non_numeric

if __name__ == "__main__":
    df = process_dataset(DATASET_PATH)

    print(instances_per_class(df))
    print(df.to_string())

    transform_non_numeric(df)
    df_correlation(df)

