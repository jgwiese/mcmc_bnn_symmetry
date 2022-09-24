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
