import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as distributions
from typing import Dict


class Regression:
    def __init__(self, transformation, parameters_prior, data_std_prior, outputs_likelihood):
        self._transformation = transformation
        self._parameters_prior = parameters_prior
        self._data_std_prior = data_std_prior
        self._outputs_likelihood = outputs_likelihood
    
    def sample(self, pm_parameters):
        #inputs = parameters["inputs"]
        #outputs = parameters["outputs"]
        inputs = pm_parameters[:, 0][:, jnp.newaxis]
        outputs = pm_parameters[:, 1][:, jnp.newaxis]
        parameters = numpyro.sample("parameters", self._parameters_prior)
        means = self._transformation.apply_from_vector(inputs=inputs, parameters_vector=parameters)
        std = numpyro.sample("log_std", self._data_std_prior)
        
        with numpyro.plate("data", size=inputs.shape[0], dim=-2):
            return numpyro.sample("outputs", self._outputs_likelihood(means, std), obs=outputs)

