import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as distributions
from typing import Dict
from flax.core import freeze
from utils.conversion import flax_parameters_dict_to_jax_parameter_vector
from tqdm import tqdm


class Regression:
    def __init__(self, transformation, dataset):
        self._transformation = transformation
        self._parameters_prior = distributions.Normal
        self._data_std_prior = distributions.HalfNormal(jnp.ones(1))
        self._outputs_likelihood = distributions.Normal
        self._dataset = dataset
        self._parameters_template = self._transformation.init(jax.random.PRNGKey(0), self._dataset[0][0])
    
    def log_posterior_parameters(self, parameters, std):
        inputs = self._dataset.data_train[:, self._dataset.conditional_indices]
        outputs = self._dataset.data_train[:, self._dataset.dependent_indices]
        n, d = parameters.shape

        parameters_prior = self._parameters_prior(jnp.zeros(d), jnp.ones(d))
        means_all = jax.vmap(self._transformation.apply_from_vector, in_axes=(None, 0))(inputs, parameters).squeeze()
        outputs_all = jnp.stack([outputs] * n).squeeze()
        log_likelihood_all = self._outputs_likelihood(means_all, std).log_prob(outputs_all).sum(-1)
        log_prior_all = parameters_prior.log_prob(parameters.reshape((-1, d))).sum(-1)
        values = log_likelihood_all + log_prior_all
        return jnp.array(values)

    def __call__(self):
        inputs = self._dataset.data_train[:, self._dataset.conditional_indices]
        outputs = self._dataset.data_train[:, self._dataset.dependent_indices]
        
        parameters_dict = {"params": {}}
        for key in self._parameters_template["params"].keys():
            layer = self._parameters_template["params"][key]
            kernel = layer["kernel"]
            bias = layer["bias"]
            parameters_dict["params"][key] = {}
            parameters_dict["params"][key]["kernel"] = numpyro.sample("{}_{}_{}".format("params", key, "kernel"), self._parameters_prior(jnp.zeros(kernel.shape), jnp.ones(kernel.shape)))
            parameters_dict["params"][key]["bias"] = numpyro.sample("{}_{}_{}".format("params", key, "bias"), self._parameters_prior(jnp.zeros(bias.shape), jnp.ones(bias.shape)))
        parameters_dict = freeze(parameters_dict)

        # flatten parameters and make a deterministic variable from it.
        parameters = numpyro.deterministic("parameters", flax_parameters_dict_to_jax_parameter_vector(parameters_dict))

        #means = self._transformation.apply(parameters, inputs)
        means = self._transformation.apply_from_vector(inputs=inputs, parameters_vector=parameters)
        std = numpyro.sample("std", self._data_std_prior)
        
        with numpyro.plate("data", size=inputs.shape[0], dim=-2):
            return numpyro.sample("outputs", self._outputs_likelihood(means, std), obs=outputs)
