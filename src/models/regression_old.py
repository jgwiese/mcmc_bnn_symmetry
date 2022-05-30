import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as distributions
from typing import Dict


class RegressionOld:
    def __init__(self, transformation, dataset):
        self._transformation = transformation
        parameters_size = transformation.parameters_size(dataset[0][0])
        self._parameters_prior = distributions.Normal(
            jnp.zeros(parameters_size),
            jnp.ones(parameters_size)
        )
        self._data_std_prior = distributions.HalfNormal(jnp.ones(1))
        self._outputs_likelihood = distributions.Normal
        self._dataset = dataset
    
    def __call__(self):
        inputs = self._dataset.data[:, self._dataset.conditional_indices]
        outputs = self._dataset.data[:, self._dataset.dependent_indices]
        parameters = numpyro.sample("parameters", self._parameters_prior)
        means = self._transformation.apply_from_vector(inputs=inputs, parameters_vector=parameters)
        std = numpyro.sample("std", self._data_std_prior)
        
        with numpyro.plate("data", size=inputs.shape[0], dim=-2):
            return numpyro.sample("outputs", self._outputs_likelihood(means, std), obs=outputs)

