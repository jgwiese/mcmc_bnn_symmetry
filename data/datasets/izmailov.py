import jax.numpy as jnp
from data.datasets import Dataset


class Izmailov(Dataset):
    def __init__(self, data=None, normalization=True):
        if data is None:
            data = jnp.load("/home/gw/data/datasets/izmailov_data.npy")
        super().__init__(data=data, normalization=normalization)
    
    def __getitem__(self, index):
        sample = self._data[index]
        return sample[:, :1], sample[:, 1:]

    def __len__(self):
        return len(self._data)

