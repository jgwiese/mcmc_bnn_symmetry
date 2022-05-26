import jax
import jax.numpy as jnp
import numpy as np
import numpyro.distributions as distributions
from data.datasets import Dataset


class Generative(Dataset):
    def __init__(self, rng_key, transformation, parameters_prior, inputs_prior, outputs_likelihood, outputs_likelihood_std, n=128, data=None, normalization="standardization"):
        parameters_key, inputs_key, outputs_key = jax.random.split(rng_key, 3)
        if data is None:
            # Generate data
            parameters_samples = parameters_prior.sample(key=parameters_key, sample_shape=(n, ))
            inputs_samples = inputs_prior.sample(key=inputs_key, sample_shape=(n, ))

            means = jax.vmap(transformation.apply_from_vector)(inputs_samples, parameters_samples).squeeze()
            outputs_distributions = outputs_likelihood(means, jnp.ones_like(means) * outputs_likelihood_std)
            outputs_samples = outputs_distributions.sample(key=outputs_key, sample_shape=(1, )).T
            data = np.concatenate([np.array(inputs_samples), np.array(outputs_samples)], axis=-1)
        super().__init__(data=np.array(data), normalization=normalization)
    
    def __getitem__(self, index):
        sample = self._data[index]
        return jnp.expand_dims(sample.T[0], axis=-1), jnp.expand_dims(sample.T[1], axis=-1)
    
    def __len__(self):
        return len(self._data)

