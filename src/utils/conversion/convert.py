import flax.linen as nn
import jax.numpy as jnp
import numpy as np
import torch
from jax.tree_util import tree_map, tree_flatten, tree_unflatten
from functools import reduce


def flax_parameters_dict_to_jax_parameter_vector(parameters_dict):
    leaves, treedef = tree_flatten(tree_map(lambda x: x.flatten(), parameters_dict))
    return jnp.concatenate(leaves)


def flax_parameters_dict_to_torch_parameters_vector(parameters_dict):
    torch_parameters = []
    for layer in parameters_dict["params"]:
        torch_parameters.append(parameters_dict["params"][layer]["kernel"].flatten(order="F"))
        torch_parameters.append(parameters_dict["params"][layer]["bias"].flatten(order="F"))
    return torch.from_numpy(np.array(jnp.concatenate(torch_parameters)))


def torch_parameters_vector_to_flax_parameters_dict(torch_parameters_vector, flax_parameters_dict_template):
    parameters_vector = jnp.array(torch_parameters_vector.detach().numpy())
    parameters = flax_parameters_dict_template
    leaves, treedef = tree_flatten(parameters)
    new_leaves = []
    accumulator = 0
    for leaf in leaves:
        size = reduce(lambda x, y: x * y, [s for s in leaf.shape])
        new_leaves.append(parameters_vector[accumulator:accumulator + size].reshape(leaf.shape, order="F"))
        accumulator += size
    new_parameters = tree_unflatten(treedef, new_leaves)
    return new_parameters


def torch_parameters_vector_to_flax_parameters_vector(torch_parameters_vector, flax_parameters_dict_template):
    torch_parameters_vector = jnp.array(torch_parameters_vector.detach().numpy())
    flax_parameters_vector = np.zeros_like(torch_parameters_vector)
    accumulator = 0
    for layer in flax_parameters_dict_template["params"]:
        layer_parameters = flax_parameters_dict_template["params"][layer]
        layer_kernel = layer_parameters["kernel"]
        layer_bias = layer_parameters["bias"]
        kernel_size = reduce(lambda x, y: x * y, [s for s in layer_kernel.shape])
        bias_size = reduce(lambda x, y: x * y, [s for s in layer_bias.shape])
        
        flax_parameters_vector[accumulator:accumulator + bias_size] = torch_parameters_vector[accumulator + kernel_size:accumulator + kernel_size + bias_size]
        flax_parameters_vector[accumulator + bias_size:accumulator + bias_size + kernel_size] = torch_parameters_vector[accumulator:accumulator + kernel_size]
        
        accumulator += kernel_size + bias_size
    return jnp.array(flax_parameters_vector)


def torch_to_flax_permutation(flax_parameters_dict_template):
    leaves, treedef = tree_flatten(flax_parameters_dict_template)
    parameters_size = sum([reduce(lambda x, y: x * y, leaf.shape) for leaf in leaves])
    permutation_indices = np.arange(parameters_size)
    accumulator = 0
    for layer in flax_parameters_dict_template["params"]:
        layer_parameters = flax_parameters_dict_template["params"][layer]
        layer_kernel = layer_parameters["kernel"]
        layer_bias = layer_parameters["bias"]
        kernel_size = reduce(lambda x, y: x * y, [s for s in layer_kernel.shape])
        bias_size = reduce(lambda x, y: x * y, [s for s in layer_bias.shape])
        
        bias_indices = permutation_indices[accumulator + kernel_size:accumulator + kernel_size + bias_size].copy()
        kernel_indices = permutation_indices[accumulator:accumulator + kernel_size].copy()

        permutation_indices[accumulator:accumulator + bias_size] = bias_indices
        permutation_indices[accumulator + bias_size:accumulator + bias_size + kernel_size] = kernel_indices

        accumulator += kernel_size + bias_size
    return permutation_indices
