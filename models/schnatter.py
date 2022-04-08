import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as distributions


class Schnatter:
    def __init__(self):
        self._parameters_prior = distributions.Normal(jnp.zeros((4, 1)), jnp.ones((4, 1)))
    
    def sample(self, observations):
        parameters = numpyro.sample("parameters", self._parameters_prior)
        normal_a_sample = numpyro.sample("normal_a", distributions.Normal(parameters[0], parameters[1]))
        normal_b_sample = numpyro.sample("normal_b", distributions.Normal(parameters[2], parameters[3]))
        x = 0.5 * normal_a_sample + 0.5 * normal_b_sample
        
        with numpyro.plate("data", size=inputs.shape[0], dim=-2):
            return numpyro.sample("outputs", self._outputs_likelihood(means, std), obs=outputs)

