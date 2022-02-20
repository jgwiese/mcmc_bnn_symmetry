from transformation import AbstractTransformation
import torch


class LinearLog(AbstractTransformation):
    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        self._bias: bool = bias
        parameters_size_tmp = in_features * out_features
        if self._bias:
            parameters_size_tmp += out_features

        super().__init__(
            domain=torch.Size([in_features]),
            image=torch.Size([out_features]),
            parameters_size=parameters_size_tmp
        )

    def forward(self, inputs, parameters, *args, **kwargs):
        # TODO: Supports only one parameter set at a time, no parameter batches
        weights = torch.exp(parameters[:self.domain[0] * self.image[0]].reshape((self.domain[0], self.image[0])))
        result = inputs @ weights
        if self._bias:
            bias = torch.exp(parameters[-self.image[0]:])
            result += bias
        return result
    
    def forward_evaluation(self, inputs, parameters, *args, **kwargs):
        weights = parameters[:self.domain[0] * self.image[0]].reshape((self.domain[0], self.image[0]))
        result = inputs @ weights
        if self._bias:
            bias = parameters[-self.image[0]:]
            result += bias
        return result
    
    @property
    def bias(self) -> bool:
        return self._bias
