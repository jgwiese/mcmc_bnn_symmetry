from transformation import AbstractTransformation, ActivationType, ACTIVATION_FUNCTIONS
import torch


class Activation(AbstractTransformation):
    def __init__(self, activation_type: ActivationType):
        super().__init__(
            domain=torch.Size(),
            image=torch.Size(),
            parameters_size=0
        )
        self._activation_type: ActivationType = activation_type
        self._activation_function = ACTIVATION_FUNCTIONS[activation_type]

    def forward(self, inputs, parameters, *args, **kwargs):
        return self._activation_function(inputs)

    @property
    def activation_type(self):
        return self._activation_type
