import os
from pathlib import Path
home = str(Path.home())


PATH_DATASETS = os.path.join(home, "data/datasets/")
PATH_THESIS = os.path.join(home, "data/experiments/master_thesis")
PATH_RESULTS = os.path.join(PATH_THESIS, "results")
PATH_FIGURES = os.path.join(PATH_THESIS, "figures")
