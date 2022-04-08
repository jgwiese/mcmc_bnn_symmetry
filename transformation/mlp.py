import jax
from jax.tree_util import tree_map, tree_flatten, tree_unflatten
import flax.linen as nn
from typing import Sequence, Callable
from functools import reduce


class MLP(nn.Module):
    features: Sequence[int]
    activation: Callable = lambda x: x
    
    @nn.compact
    def __call__(self, x):
        for i, feature in enumerate(self.features[:-1]):
            x = self.activation(nn.Dense(feature, use_bias=True, name=f"dense_{i}")(x))
        x = nn.Dense(self.features[-1], name=f"dense_{len(self.features) - 1}")(x)
        return x
    
    def parameters_size(self, inputs):
        parameters = self.init(jax.random.PRNGKey(0), inputs)
        leaves, treedef = tree_flatten(parameters)
        l = sum([reduce(lambda x, y: x * y, leaf.shape) for leaf in leaves])
        return l
    
    def init_from_vector(self, inputs, parameters_vector):
        parameters = self.init(jax.random.PRNGKey(0), inputs)  # I do not need random behaviour here.
        leaves, treedef = tree_flatten(parameters)
        new_leaves = []
        accumulator = 0
        for leaf in leaves:
            size = reduce(lambda x, y: x * y, [s for s in leaf.shape])
            new_leaves.append(parameters_vector[accumulator:accumulator + size].reshape(leaf.shape))
            accumulator += size
        new_parameters = tree_unflatten(treedef, new_leaves)
        return new_parameters
    
    def apply_from_vector(self, inputs, parameters_vector):
        parameters = self.init_from_vector(inputs=inputs, parameters_vector=parameters_vector)
        return self.apply(parameters, inputs)

