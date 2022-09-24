from data import shift_scale
from abc import ABC, abstractmethod
import jax
import jax.numpy as jnp


split_default = {
    "data_train": [],
    "data_validate": [],
    "data_test": []
}

class Dataset(ABC):
    def __init__(self, data, normalization, split: dict = None):
        super().__init__()
        self._data = data
        self._split = split
        if self._split is None:
            self._split = split_default
            self._split["data_train"] = list(range(len(self._data)))
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
    def split(self):
        return self._split
    
    @property
    def data(self):
        return self._data
    
    @property
    def data_train(self):
        return self._data[jnp.array(self._split["data_train"])]
    
    @property
    def data_validate(self):
        return self._data[jnp.array(self._split["data_validate"])]
    
    @property
    def data_test(self):
        return self._data[jnp.array(self._split["data_test"])]

    @property
    def std(self):
        return self._std

    @property
    def mean(self):
        return self._mean

