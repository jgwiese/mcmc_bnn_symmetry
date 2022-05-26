import jax.numpy as jnp


def shift_scale(data, scale, shift):
    return (data - shift) / scale


def standardize(data):
    std, mean = jnp.std(data, axis=0), jnp.mean(data, axis=0)
    return scale(data, std, mean)

