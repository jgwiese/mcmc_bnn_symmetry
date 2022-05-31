from numpyro import distributions
import jax.numpy as jnp
import numpy as np
from utils import statistics


class ChainEstimator:
    """
    modes: number of assumed non-symmetric modes that provide diversity
    p: desired bounded probability to visit all non-symmetric modes
    """
    def __init__(self, modes, p):
        self._modes = modes
        self._p = p
    
    def number_of_chains(self):
        probabilities = np.exp(distributions.HalfNormal(1.0).log_prob(jnp.linspace(0.0, 1.96, self._modes)))
        expected_value = statistics.expected_number_of_tickets_nonuniform(self._modes, probabilities)
        return int(np.ceil(statistics.markov_inequality(expected_value, self._p)))
