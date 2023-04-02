import jax.numpy as jnp
import math

"""
def rmse(y_true, y_pred):
    return jnp.power(jnp.power(y_true - y_pred, 2).mean(), 0.5)


def nll_gaussian(y_true, y_pred, sigma):
    N = len(y_true)
    return (0.5 * N) * jnp.log(2 * math.pi) + N * jnp.log(sigma) + (0.5 / jnp.power(sigma, 2)) * jnp.power(y_true - y_pred, 2).sum()

def lppd():
    pass
"""
