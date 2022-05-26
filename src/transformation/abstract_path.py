from transformation import AbstractTransformation
import torch


# TODO: Strictly speaking this is a Transformation also, should be converted anytime soon.
class AbstractPath(AbstractTransformation):
    def __init__(self):
        self._loss_history = {}
        super().__init__(
            domain=torch.Size([1]),
            image=torch.Size([1]),
            parameters_size=1
        )

