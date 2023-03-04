import global_settings
import models
from utils import experiments
from utils.experiments import settings


class ExperimentSampleStandard(experiments.AbstractExperimentSample):
    def __init__(self, settings: settings.SettingsExperimentSample):
        super().__init__(settings=settings)

    def _load_model(self):
        if self._settings.dataset in global_settings.DATASET_NAMES_BENCHMARK or self._settings.dataset in global_settings.DATASET_NAMES_TOY:
            return models.Regression(transformation=self._model_transformation, dataset=self._dataset)
