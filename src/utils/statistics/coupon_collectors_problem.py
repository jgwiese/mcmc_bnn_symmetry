import jax.numpy as jnp
from itertools import combinations


def expected_number_of_tickets(n):
    """ Coupon collector's problem. For large n I should use the approximation. Each ticket of the n tickets has the same probability. """
    harmonics = jnp.sum(1.0 / jnp.arange(1, n + 1))
    return n * harmonics


def expected_number_of_tickets_nonuniform(m, p):
    """ version for nonuniformly distributed tickets """
    assert m == p.shape[0], f"p needs {m} entries"
    p = p / p.sum()
    #print("normalized probabilities: {}".format(p))

    total_sum = 0.0
    for q in range(m):
        factor = jnp.power(jnp.array([-1.0]), m - 1 - q).squeeze()
        indices_sets = list(combinations(list(range(m)), q))
        inner_sum = 0.0
        for element in indices_sets:
            inner_sum += 1.0 / (1.0 - p[list(element)].sum())
        total_sum += factor * inner_sum
    return total_sum


def markov_inequality(expected_value, p):
    """ Implements the Markov inequality. p is the desired probability for the upper bound of the random variable. for the coupon collector's problem
    this equals to the desired probability of observing all tickets when drawing the calculated number of tickets. """
    assert 0.0 <= p and p <= 1.0
    c = 1.0 / (1.0 - p)
    return c * expected_value

