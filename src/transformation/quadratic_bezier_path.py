from transformation import AbstractPath
import torch


class QuadraticBezierPath(AbstractPath):
    def __init__(self, a, b):
        super().__init__()
        self._a = a
        self._b = b
    
    def __call__(self, inputs, parameters):
        return self.forward(inputs=inputs, parameters=parameters)
    
    def forward(self, inputs, parameters):
        outputs = torch.pow(1.0 - inputs, 2) * self._a + 2.0 * inputs * (1 - inputs) * parameters + torch.pow(inputs, 2) * self._b
        return outputs
    
    @property
    def a(self):
        return self._a
    
    @property
    def b(self):
        return self._b
