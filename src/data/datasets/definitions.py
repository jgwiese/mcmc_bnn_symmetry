import utils
from data import datasets


def create_dataset(settings: utils.SettingsExperiment):
    if settings.dataset == "izmailov":
        dataset = datasets.Izmailov(normalization=settings.dataset_normalization)
    elif settings.dataset == "sinusoidal":
        dataset = datasets.Sinusoidal(
            normalization=settings.dataset_normalization,
        )
    elif settings.dataset == "regression2d":
        dataset = datasets.Regression2d(
            normalization=settings.dataset_normalization,
        )
    else:
        print("unknown dataset")
        dataset = None
    return dataset

