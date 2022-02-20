from transformation import AbstractTransformation
import torch


class Constant(AbstractTransformation, torch.nn.Module):
    def __init__(self, value: torch.Tensor):
        super().__init__(domain=value.shape, image=value.shape, parameters_size=0)
        self._value = value

    def forward(self, inputs, parameters):
        return torch.stack([self._value] * inputs.shape[0], dim=0)
