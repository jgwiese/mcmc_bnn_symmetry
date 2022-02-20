import jax
import jax.numpy as jnp
from data.datasets import Dataset


class Sinusoidal(Dataset):
    def __init__(self, data=None, n=150, x_lower=0.0, x_upper=8.0, sigma_noise=0.3, normalization=True, rng_key=None):
        if data is None:
            assert rng_key is not None, "random key needed"
            x_key, y_key = jax.random.split(rng_key)
            x = jax.random.uniform(x_key, shape=(n,)) * (x_upper - x_lower)
            noise = jax.random.normal(key=y_key, shape=x.shape) * sigma_noise
            y = jnp.sin(x) + noise
            data = jnp.stack([x, y], axis=-1)
        super().__init__(data=data, normalization=normalization)

    def __getitem__(self, index):
        sample = self._data[index]
        return jnp.expand_dims(sample.T[0], -1), jnp.expand_dims(sample.T[1], -1)

    def __len__(self):
        return len(self._data)

