from typing import Any, Callable, Sequence
from flax.linen.module import Module
import jax
from jax.tree_util import tree_map, tree_flatten, tree_unflatten, tree_multimap
from functools import reduce


class Sequential(Module):
    layers: Sequence[Callable[..., Any]]

    def __call__(self, *args, **kwargs):
        if not self.layers:
            raise ValueError(f'Empty Sequential module {self.name}.')

        outputs = self.layers[0](*args, **kwargs)
        for layer in self.layers[1:]:
            outputs = layer(outputs)
        return outputs

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

