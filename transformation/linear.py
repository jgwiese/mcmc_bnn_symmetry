from transformation import AbstractTransformation
import jax.numpy as jnp


class Linear(AbstractTransformation):
    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        self._bias: bool = bias
        parameters_size = in_features * out_features
        if self._bias:
            parameters_size += out_features

        super().__init__(
            domain=in_features,
            image=out_features,
            parameters_size=parameters_size
        )

    def forward(self, inputs, parameters, *args, **kwargs):
        w = parameters[:self.domain * self.image].reshape((self.image, self.domain))
        result = jnp.matmul(inputs, w)
        if self._bias:
            b = parameters[-self.image:]
            result = jnp.add(result, b)
        return result
    
    @property
    def bias(self) -> bool:
        return self._bias

