import jax.numpy as jnp
from data.datasets import ConditionalDataset
import os
import global_settings


class Izmailov(ConditionalDataset):
    def __init__(self, normalization="standardization", split: dict = None):
        data = jnp.load(os.path.join(global_settings.PATH_DATASETS, "izmailov_data.npy"))
        super().__init__(
            data=data,
            normalization=normalization,
            conditional_indices=[0],
            dependent_indices=[1],
            split=split
        )

