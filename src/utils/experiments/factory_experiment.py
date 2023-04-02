from utils import experiments


class FactoryExperiment:
    def __init__(self, key, **kwargs):
        self._classes = {
            "ExperimentSampleStandard": experiments.ExperimentSampleStandard,
        }
        self._key = key
        self._kwargs = kwargs
    
    def __call__(self):
        return self._classes[self._key](**self._kwargs)

