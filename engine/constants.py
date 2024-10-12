import os.path
from pathlib import Path

ROOT_DIR_PATH = Path(__file__).parent.parent.resolve()
DATASET_DIR_NAME = "data"
DATASET_FILE_NAME = "dataset.csv"
DATASET_PATH = os.path.join(ROOT_DIR_PATH, DATASET_DIR_NAME, DATASET_FILE_NAME)
