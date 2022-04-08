import jax.numpy as jnp


def expected_number_of_chains(n):
    """ Coupon collector's problem. For large n I should use the approximation. """
    harmonics = jnp.sum(1.0 / jnp.arange(1, n + 1))
    return n * harmonics


def bounded_expected_number_of_chains(n, p):
    """ Implements the Markov inequality. """
    assert 0.0 <= p and p <= 1.0
    c = 1.0 / (1.0 - p)
    return c * expected_number_of_chains(n=n)
    
