from enum import Enum
from typing import Dict, Any
from transformation import Identity
import torch


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
