from engine.constants import DATASET_PATH
from engine.processing import process_dataset
from engine.statistics import instances_per_class

if __name__ == "__main__":
    df = process_dataset(DATASET_PATH)

    print(instances_per_class(df))
    print(df.to_string())
