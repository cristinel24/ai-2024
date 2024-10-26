import os.path
from pathlib import Path

ROOT_DIR_PATH = Path(__file__).parent.parent.resolve()

DATASET_DIR_NAME = "data"
DATASET_FILE_NAME = "dataset.csv"
DATASET_PATH = os.path.join(ROOT_DIR_PATH, DATASET_DIR_NAME, DATASET_FILE_NAME)

PLOTS_DIR_NAME = "plots"
PLOTS_PATH = os.path.join(ROOT_DIR_PATH, PLOTS_DIR_NAME)
if not os.path.exists(PLOTS_PATH):
    os.mkdir(PLOTS_PATH)
