import jax.numpy as jnp


def full_pca(data):
    n, dim = data.shape
    mean = data.mean(0)
    covariance = (1.0 / n) * data.T @ data - jnp.outer(mean, mean)
    values, vectors = jnp.linalg.eig(covariance)

    # convert from complex to float
    values = jnp.array(values, dtype=float)
    #indices = jnp.argsort(values)
    #values = jnp.flip(values[indices])
    #vectors = jnp.flip(vectors[indices])
    return values, vectors

def spread(data):
    values, vectors = full_pca(data=data)
    return values.mean()
