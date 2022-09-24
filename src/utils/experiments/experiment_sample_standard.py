import models
from utils import settings
from utils import experiments


dataset_names_benchmark = [
    'airfoil',
    'boston',
    'concrete',
    'diabetes',
    'energy',
    'forest_fire',
    'wine',
    'yacht'
]


class ExperimentSampleStandard(experiments.AbstractExperimentSample):
    def __init__(self, settings: settings.SettingsExperimentSample):
        super().__init__(settings=settings)

    def _load_model(self):
        if self._settings.dataset == "izmailov" or self._settings.dataset == "sinusoidal" or self._settings.dataset == "regression2d" or self._settings.dataset in dataset_names_benchmark:
            return models.Regression(transformation=self._model_transformation, dataset=self._dataset)
