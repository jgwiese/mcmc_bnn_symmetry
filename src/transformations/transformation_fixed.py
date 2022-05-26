from transformation import AbstractTransformation
import torch


class TransformationFixed(AbstractTransformation):
    def __init__(self, transformation):
        super().__init__(
            domain=transformation.domain,
            image=transformation.image,
            parameters_size=transformation.parameters_size
        )
        self._transformation = transformation
        self._estimator = (torch.randn(transformation.parameters_size) * 0.001).requires_grad_(True)
    
    def __call__(self, inputs, parameters=None):
        return self.forward(inputs=inputs, parameters=parameters)
    
    def forward(self, inputs, parameters=None):
        if parameters is None:
            parameters = self._estimator
        return self._transformation.forward(inputs=inputs, parameters=parameters)
    
    @property
    def transformation(self):
        return self._transformation
    
    @property
    def estimator(self):
        return self._estimator
