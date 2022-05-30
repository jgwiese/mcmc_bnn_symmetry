import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as distributions
from typing import Dict
from flax.core import freeze
from utils.conversion import flax_parameters_dict_to_jax_parameter_vector


class RegressionConstrained:
    def __init__(self, transformation, dataset):
        self._transformation = transformation
        self._parameters_prior_constrained = distributions.HalfNormal
        self._parameters_prior = distributions.Normal
        self._data_std_prior = distributions.HalfNormal(jnp.ones(1))
        self._outputs_likelihood = distributions.Normal
        self._dataset = dataset
        self._parameters_template = self._transformation.init(jax.random.PRNGKey(0), self._dataset[0][0])
        self._layer_keys = list(self._parameters_template["params"].keys())
    
    def __call__(self):
        inputs = self._dataset.data[:, self._dataset.conditional_indices]
        outputs = self._dataset.data[:, self._dataset.dependent_indices]
        
        parameters_dict = {"params": {}}
        for i, key in enumerate(self._layer_keys):
            layer = self._parameters_template["params"][key]
            kernel = layer["kernel"]
            bias = layer["bias"]
            parameters_dict["params"][key] = {}
            parameters_dict["params"][key]["kernel"] = numpyro.sample("{}_{}_{}".format("params", key, "kernel"), self._parameters_prior(jnp.zeros(kernel.shape), jnp.ones(kernel.shape)))

            if i < len(self._layer_keys):
                bias_differences = numpyro.sample("{}_{}_{}_differences".format("params", key, "bias"), self._parameters_prior_constrained(jnp.ones(bias.shape)))
                parameters_dict["params"][key]["bias"] = numpyro.deterministic("{}_{}_{}".format("params", key, "bias"), jnp.cumsum(bias_differences))
            else:
                parameters_dict["params"][key]["bias"] = numpyro.sample("{}_{}_{}".format("params", key, "bias"), self._parameters_prior(jnp.zeros(bias.shape), jnp.ones(bias.shape)))

        parameters_dict = freeze(parameters_dict)

        # flatten parameters and make a deterministic variable from it.
        parameters = numpyro.deterministic("parameters", flax_parameters_dict_to_jax_parameter_vector(parameters_dict))

        means = self._transformation.apply(parameters_dict, inputs)
        #means = self._transformation.apply_from_vector(inputs=inputs, parameters_vector=parameters)
        std = numpyro.sample("std", self._data_std_prior)
        
        with numpyro.plate("data", size=inputs.shape[0], dim=-2):
            return numpyro.sample("outputs", self._outputs_likelihood(means, std), obs=outputs)
