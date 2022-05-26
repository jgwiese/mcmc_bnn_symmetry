from enum import Enum
from typing import Dict, Any
from transformation import Identity
import torch
import utils
import flax.linen as nn
import transformation


def create_model_transformation(settings: utils.SettingsExperiment, dataset):
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


class ActivationType(Enum):
    """
    Enumeration for supported activation functions.
    """

    SIGMOID = 1
    TANH = 2
    RELU = 3
    SELU = 4
    IDENTITY = 5


ACTIVATION_FUNCTIONS: Dict[ActivationType, Any] = {
    ActivationType.SIGMOID: torch.sigmoid,
    ActivationType.TANH: torch.tanh,
    ActivationType.RELU: torch.relu,
    ActivationType.SELU: torch.selu,
    ActivationType.IDENTITY: Identity
}
