import jax
import jax.numpy as jnp
from data.datasets import Dataset
from transformations import Sequential
import flax.linen as nn


class IRMAE(Dataset):
    def __init__(self, rng_key=None, data=None, n=150, latent_dim=1, output_dim=1, sigma_noise=0.1, normalization=True):
        if data is None:
            assert rng_key is not None, "random key needed"
            transformation = Sequential([nn.Dense(output_dim), nn.tanh])
            z_key, init_key, noise_key = jax.random.split(rng_key, 3)
            z = jax.random.normal(z_key, shape=(n, latent_dim)) * 8
            x, parameters = transformation.init_with_output(init_key, z)
            noise = jax.random.normal(key=noise_key, shape=x.shape) * sigma_noise
            x = x + noise
            data = x
        super().__init__(data=data, normalization=normalization)

    def __getitem__(self, index):
        sample = self._data[index]
        return sample

    def __len__(self):
        return len(self._data)

