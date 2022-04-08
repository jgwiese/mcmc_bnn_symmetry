from data import scale
from abc import ABC, abstractmethod
import jax
import jax.numpy as jnp


# TODO: I should probably use torch datasets
class Dataset(ABC):
    def __init__(self, data, normalization):
        super().__init__()
        self._data = data
        self._normalization = normalization
        self._std, self._mean = jnp.std(self._data, axis=0), jnp.mean(self._data, axis=0)
        
        if normalization == "standardization":
            self._data = scale(self._data, self._std, self._mean)
        elif normalization == "none":
            pass
        else:
            pass
    
    def split(self, rng_key, ratio):
        assert ratio >= 0.0 and ratio <= 1.0
        data_shuffled = jax.random.permutation(rng_key, self._data, axis=0)
        data_train = data_shuffled[:int(ratio * len(data_shuffled))].clone()
        data_validate = data_shuffled[int(ratio * len(data_shuffled)):].clone()
        
        # rescale data if necessary
        if self._normalization == "standardization":
            data_train = self._mean + self._std * data_train
            data_validate = self._mean + self._std * data_validate
        elif self._normalization == "none":
            pass
        else:
            pass

        # separate into two new datasets
        dataset_train = self.__class__(
            data=data_train,
            normalization=self._normalization
        )
        dataset_validate = self.__class__(
            data=data_validate,
            normalization=self._normalization
        )
        return dataset_train, dataset_validate
    
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

