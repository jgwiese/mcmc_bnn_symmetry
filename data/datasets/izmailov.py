import jax.numpy as jnp
from data.datasets import Dataset


class Izmailov(Dataset):
    def __init__(self, data=None, normalization=True):
        if data is None:
            data = jnp.load("/home/gw/data/datasets/izmailov_data.npy")
        super().__init__(data=data, normalization=normalization)
    
    def __getitem__(self, index):
        sample = self._data[index]
        conditional = sample[..., :1]
        dependent = sample[..., 1:]
        return conditional, dependent

    def __len__(self):
        return len(self._data)

