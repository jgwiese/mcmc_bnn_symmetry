import utils
from data import datasets
import transformation
import models


def create_model(settings: utils.SettingsExperiment, dataset: datasets.Dataset, model_transformation: transformation.Sequential):
    model = None
    if settings.dataset == "izmailov" or settings.dataset == "sinusoidal" or settings.dataset == "regression2d":
        model = models.Regression(transformation=model_transformation, dataset=dataset)
    return model

