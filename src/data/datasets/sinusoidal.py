import jax
import jax.numpy as jnp
from data.datasets import ConditionalDataset


class Sinusoidal(ConditionalDataset):
    def __init__(self, n=150, x_lower=0.0, x_upper=8.0, sigma_noise=0.3, normalization="standardization", rng_key=jax.random.PRNGKey(0)):
        x_key, y_key = jax.random.split(rng_key)
        x = jax.random.uniform(x_key, shape=(n,)) * (x_upper - x_lower)
        noise = jax.random.normal(key=y_key, shape=x.shape) * sigma_noise
        y = jnp.sin(x) + noise
        data = jnp.stack([x, y], axis=-1)
        super().__init__(
            data=data,
            normalization=normalization,
            conditional_indices=[0],
            dependent_indices=[1]
        )

