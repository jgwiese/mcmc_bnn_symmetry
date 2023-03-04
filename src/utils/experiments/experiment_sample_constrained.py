import models
from utils import experiments
from utils.experiments import settings


class ExperimentSampleConstrained(experiments.AbstractExperimentSample):
    def __init__(self, settings: settings.SettingsExperimentSample):
        super().__init__(settings=settings)

    def _load_model(self):
        if self._settings.dataset == "izmailov" or self._settings.dataset == "sinusoidal" or self._settings.dataset == "regression2d":
            return models.RegressionConstrained(transformation=self._model_transformation, dataset=self._dataset)
