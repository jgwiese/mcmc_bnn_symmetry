from transformation import AbstractTransformation
import torch


class Identity(AbstractTransformation):
    def __init__(self):
        super().__init__(
            domain=torch.Size(),
            image=torch.Size(),
            parameters_size=0
        )

    def forward(self, inputs, parameters, *args, **kwargs):
        return inputs
