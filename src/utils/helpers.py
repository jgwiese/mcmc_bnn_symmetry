import utils
import flax.linen as nn
import transformations
import models
from data import datasets
from utils import settings


def create_dataset(settings: settings.SettingsExperiment):
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


def create_model_transformation(settings: settings.SettingsExperiment, dataset):
    layers = []
    for l in range(settings.hidden_layers):
        layers.append(nn.Dense(settings.hidden_neurons))
        if settings.activation == "tanh":
            layers.append(nn.tanh)
    
    layers.append(nn.Dense(len(dataset.dependent_indices)))
    if settings.activation_last_layer == "tanh":
        layers.append(nn.tanh)
    
    model_transformation = transformation.Sequential(layers)
    return model_transformation


def create_model(settings: settings.SettingsExperiment, dataset: datasets.Dataset, model_transformation: transformations.Sequential):
    model = None
    if settings.dataset == "izmailov" or settings.dataset == "sinusoidal" or settings.dataset == "regression2d":
        model = models.Regression(transformation=model_transformation, dataset=dataset)
    return model

