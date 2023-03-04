from data import datasets
import models
import transformations
import numpyro
from flax import linen as nn
import transformations
import jax
from tqdm import tqdm
import jax.numpy as jnp
from utils.results import ResultSample
import datetime
import hashlib
from utils.experiments import settings
from abc import ABC, abstractmethod
from typing import Any
import os
import json
import global_settings
from utils import statistics


class AbstractExperimentSample(ABC):
    def __init__(self, settings: settings.SettingsExperiment):
        self._date = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self._settings = settings
        self._dataset = self._load_dataset()
        self._model_transformation = self._load_model_transformation()
        self._model = self._load_model()
        self._sample_statistics = self._load_statistics()
        self._kernel, self._mcmc = self._load_sampler()
    
    @abstractmethod
    def _load_model(self):
        raise NotImplementedError
    
    def _load_statistics(self):
        if self._settings.overwrite_chains is not None:
            return {
            "num_chains": self._settings.overwrite_chains
        }
        else:
            ce = statistics.ChainEstimator(modes=self._settings.identifiable_modes, p=self._settings.statistic_p)
            number_of_chains = ce.number_of_chains()
            return {
                "num_chains": number_of_chains
            }
    
    def _load_dataset(self): # this is a mess
        # load split indices
        if self._settings.dataset in global_settings.DATASET_NAMES_TOY:
            with open(os.path.join(global_settings.PATH_DATASETS, "toy_dataset_indices_0.2.json"), 'r') as f:
                indices = json.load(f)
                split = {
                    "data_train": indices[self._settings.dataset]["train"],
                    "data_validate": [],
                    "data_test": indices[self._settings.dataset]["validate"] # validate as test data, since we do not need validation data...
                }
        elif self._settings.dataset in global_settings.DATASET_NAMES_BENCHMARK:
            with open(os.path.join(os.path.join(global_settings.PATH_DATASETS, "benchmark_data"), "dataset_indices_0.2.json"), 'r') as f:
                indices = json.load(f)
                split = {
                    "data_train": indices[self._settings.dataset]["train"],
                    "data_validate": [],
                    "data_test": indices[self._settings.dataset]["validate"] # validate as test data, since we do not need validation data...
                }
        else:
            return None

        # load dataset
        if self._settings.dataset == "izmailov":
            dataset = datasets.Izmailov(normalization=self._settings.dataset_normalization, split=split)
        elif self._settings.dataset == "sinusoidal":
            dataset = datasets.Sinusoidal(normalization=self._settings.dataset_normalization, split=split)
        elif self._settings.dataset == "regression2d":
            dataset = datasets.Regression2d(normalization=self._settings.dataset_normalization, split=split)
        elif self._settings.dataset in global_settings.DATASET_NAMES_BENCHMARK:
            dataset = datasets.GenericBenchmark(dataset_name=self._settings.dataset, normalization=self._settings.dataset_normalization, split=split)
        else:
            return None
        
        return dataset

    def _load_model_transformation(self):
        layers = []
        for l in range(self._settings.hidden_layers):
            layers.append(nn.Dense(self._settings.hidden_neurons))
            if self._settings.activation == "tanh":
                layers.append(nn.tanh)
        
        layers.append(nn.Dense(len(self._dataset.dependent_indices)))
        if self._settings.activation_last_layer == "tanh":
            layers.append(nn.tanh)
        return transformations.Sequential(layers)
    
    @staticmethod
    def load_model_transformation(settings: settings.SettingsExperimentSample, dataset: Any):
        layers = []
        for l in range(settings.hidden_layers):
            layers.append(nn.Dense(settings.hidden_neurons))
            if settings.activation == "tanh":
                layers.append(nn.tanh)
        
        layers.append(nn.Dense(len(dataset.dependent_indices)))
        if settings.activation_last_layer == "tanh":
            layers.append(nn.tanh)
        return transformations.Sequential(layers)

    def _load_sampler(self):
        kernel = numpyro.infer.NUTS(self._model)
        mcmc = numpyro.infer.MCMC(
            kernel,
            num_warmup=self._settings.num_warmup,
            num_samples=self._settings.samples_per_chain,
            num_chains=1,
            progress_bar=False # Change back
        )
        return kernel, mcmc
    
    def sample(self, key):
        self._mcmc.run(key)
        return self._mcmc.get_samples()
    
    def run(self):
        print("model transformation parameters {}".format(self._model_transformation.parameters_size(self._dataset[0][0])))
        print("number of chains: {}".format(self._sample_statistics["num_chains"]))
        samples_parallel = []
        rng_key, *sub_keys = jax.random.split(jax.random.PRNGKey(self._settings.seed), self._sample_statistics["num_chains"] + 1)
        for i in tqdm(range(self._sample_statistics["num_chains"])):
            samples_parallel.append(self.sample(sub_keys[i]))
        
        samples = {}
        for samples_run in samples_parallel:
            for key in samples_run.keys():
                if key not in samples:
                    samples[key] = samples_run[key]
                else:
                    samples[key] = jnp.concatenate([samples[key], samples_run[key]])
        self._samples = samples

    def result(self):
        identifier = hashlib.md5(self._date.encode("utf-8")).hexdigest()
        result = ResultSample(
            identifier=identifier,
            date=self._date,
            experiment_type=self.__class__.__name__,
            settings=self._settings,
            dataset=self._dataset,
            indices_train=self._dataset.split["data_train"],
            indices_validate=self._dataset.split["data_validate"],
            indices_test=self._dataset.split["data_test"],
            samples=self._samples
        )
        return result

    def save(self):
        # gets an id
        result = self.result()
        return result.save()
