from data import shift_scale
from abc import ABC, abstractmethod
import jax
import jax.numpy as jnp


class Dataset(ABC):
    def __init__(self, data, normalization):
        super().__init__()
        self._data = data
        self._normalization = normalization
        self._std, self._mean = jnp.std(self._data, axis=0), jnp.mean(self._data, axis=0)
        
        if normalization == "standardization":
            self._data = shift_scale(self._data, self._std, self._mean)
        else:
            pass
    
    @abstractmethod
    def __len__(self):
        raise NotImplementedError
    
    @abstractmethod
    def __getitem__(self, index, *args, **kwargs):
        raise NotImplementedError
    
    @property
    def data(self):
        return self._data
    
    @property
    def std(self):
        return self._std

    @property
    def mean(self):
        return self._mean

