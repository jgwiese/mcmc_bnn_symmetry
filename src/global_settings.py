import os
from pathlib import Path
home = str(Path.home())


PATH_DATASETS = os.path.join(home, "data/datasets/")
PATH_THESIS = os.path.join(home, "data/experiments/master_thesis")
PATH_RESULTS = os.path.join(PATH_THESIS, "results")
PATH_RESULTS_FIXED = os.path.join(PATH_THESIS, "results_fixed")
PATH_FIGURES = os.path.join(PATH_THESIS, "figures")

# paper related
PATH_PAPER = os.path.join(home, "data/experiments/paper")
PATH_PAPER_RESULTS = os.path.join(PATH_PAPER, "results")


DATASET_NAMES_TOY = [
    "sinusoidal",
    "izmailov",
    "regression2d"
]

DATASET_NAMES_BENCHMARK = [
    "airfoil",
    "concrete",
    "diabetes",
    "energy",
    "forest_fire",
    "yacht"
]

DATASET_NAMES = {
    0: DATASET_NAMES_TOY[0],
    1: DATASET_NAMES_TOY[1],
    2: DATASET_NAMES_TOY[2],
    3: DATASET_NAMES_BENCHMARK[0],
    4: DATASET_NAMES_BENCHMARK[1],
    5: DATASET_NAMES_BENCHMARK[2],
    6: DATASET_NAMES_BENCHMARK[3],
    7: DATASET_NAMES_BENCHMARK[4],
    8: DATASET_NAMES_BENCHMARK[5],
}

