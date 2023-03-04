import numpy
import jax.numpy as jnp
import pandas
import os
import global_settings
from data.datasets import ConditionalDataset


class GenericBenchmark(ConditionalDataset):
    def __init__(self, dataset_name: str, normalization: str = "standardization", split: dict = None):
        self._dataset_name = dataset_name
        raw_data = []
        with open(os.path.join(os.path.join(global_settings.PATH_DATASETS, "benchmark_data"), f"{dataset_name}.data"), 'r') as f:
            lines = f.readlines()
            for line in lines:
                raw_data_line = jnp.asarray([float(w) for w in line.split(' ')])
                raw_data.append(raw_data_line)
        data = jnp.asarray(raw_data)
        features = len(data[0])
        super().__init__(
            data=data,
            normalization=normalization,
            conditional_indices=list(range(features - 1)),
            dependent_indices=[features - 1],
            split=split
        )

    @property
    def dataset_name(self):
        return self._dataset_name

