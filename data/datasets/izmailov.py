import jax.numpy as jnp
from data.datasets import ConditionalDataset


class Izmailov(ConditionalDataset):
    def __init__(self, normalization="standardization"):
        data = jnp.load("/home/gw/data/datasets/izmailov_data.npy")
        super().__init__(
            data=data,
            normalization=normalization,
            conditional_indices=[0],
            dependent_indices=[1]
        )

