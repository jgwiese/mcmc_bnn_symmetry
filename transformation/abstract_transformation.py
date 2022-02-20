from abc import ABC, abstractmethod


class AbstractTransformation(ABC):
    def __init__(self, domain, image, parameters_size, *args, **kwargs):
        super().__init__()
        self._domain: int = domain
        self._image: int = image
        self._parameters_size: int = parameters_size
    
    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    @property
    def domain(self) -> int:
        return self._domain

    @property
    def image(self) -> int:
        return self._image

    @property
    def parameters_size(self) -> int:
        return self._parameters_size

    @abstractmethod
    def forward(self, inputs, parameters):
        raise NotImplementedError
