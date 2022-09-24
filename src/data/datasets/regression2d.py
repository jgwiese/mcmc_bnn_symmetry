import jax
import jax.numpy as jnp
from data.datasets import ConditionalDataset


def f(x):
    return x[:, 0][:, jnp.newaxis] * (jnp.sin(x[:, 0])[:, jnp.newaxis] + jnp.cos(x[:, 1])[:, jnp.newaxis])


class Regression2d(ConditionalDataset):
    def __init__(self, n=256, sigma_noise=0.1, normalization="standardization", rng_key=jax.random.PRNGKey(0), split: dict = None):
        x_key, y_key = jax.random.split(rng_key)
        x = jax.random.uniform(x_key, shape=(n, 2)) * 4 - 2
        noise = jax.random.normal(y_key, shape=(n, 1)) * sigma_noise
        y = f(x) + noise
        data = jnp.concatenate([x, y], axis=-1)
        super().__init__(
            data=data,
            normalization=normalization,
            conditional_indices=[0, 1],
            dependent_indices=[2],
            split=split
        )

