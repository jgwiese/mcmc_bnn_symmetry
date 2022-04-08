from functools import reduce
from itertools import product, permutations
import numpy as np
from jax.tree_util import tree_map, tree_leaves, tree_flatten, tree_unflatten
import jax.numpy as jnp


class MLPSymmetryHelper:
    # so far only works for single hidden layer MLPs
    def __init__(self, parameters_template, activation_function):
        self._parameters_shapes = tree_map(lambda x: jnp.array(x.shape), parameters_template)
        self._activation_function = activation_function
    
    def symmetries_size(self):
        size = 1
        layer_keys = list(self._parameters_shapes["params"].keys())
        for i, layer_key in enumerate(layer_keys[:-1]):
            m = self._parameters_shapes["params"][layer_key]["bias"][0]
            size *= jnp.product(jnp.arange(1, m + 1)) * jnp.power(2, m)
        return size
    
    def remove_permutation_symmetries(self, parameters_vectors, bias=True):
        # DOES THIS WORK FOR MULTIPLE LAYERS? Sorting all biases at once.
        result_parameters_vectors = np.array(parameters_vectors.copy())
        # Test/separate along the kernel parameters
        parameters_size = sum(tree_leaves(tree_map(lambda leaf: reduce(lambda a, b: a * b, leaf), self._parameters_shapes)))
        leaves, treedef = tree_flatten(self._parameters_shapes)
        
        # construct tree of parameters indices
        leaves_indices = []
        i = 0
        for leaf in leaves:
            size = reduce(lambda a, b: a * b, leaf)
            leaves_indices.append(np.arange(parameters_size)[i:i + size].reshape(leaf))
            i += size
        parameters_indices = tree_unflatten(treedef, leaves_indices)
        
        sorting_indices = []
        permutation_indices = []
        layer_keys = list(self._parameters_shapes["params"].keys())
        for i, layer_key in enumerate(layer_keys[:-1]):
            layer_indices = parameters_indices["params"][layer_keys[i]]
            following_layer_indices = parameters_indices["params"][layer_keys[i + 1]]
            bias_indices = layer_indices["bias"]
            kernel_indices = layer_indices["kernel"]
            following_kernel_indices = following_layer_indices["kernel"]
            # TODO: Seems like this does not do the trick to permute the kernel rows
            if bias:
                sorting_indices.append(bias_indices.flatten())
            else:
                sorting_indices.append(kernel_indices.flatten())
            permutation_indices.append([bias_indices, kernel_indices, following_kernel_indices])

        for si, pi in zip(sorting_indices, permutation_indices):
            for i in range(len(result_parameters_vectors)):
                result_parameters_vector = result_parameters_vectors[i]

                # TODO: Only works for MLP of the shapes: [1, N, 1]
                bias_indices = pi[0].flatten()
                kernel_indices = pi[1].flatten()
                following_kernel_indices = pi[2].flatten()
                
                indices_sorted = np.argsort(result_parameters_vector[si])
                result_parameters_vector[bias_indices] = result_parameters_vector[bias_indices][indices_sorted]
                result_parameters_vector[kernel_indices] = result_parameters_vector[kernel_indices][indices_sorted]
                result_parameters_vector[following_kernel_indices] = result_parameters_vector[following_kernel_indices][indices_sorted]
                result_parameters_vectors[i] = result_parameters_vector
        return result_parameters_vectors

    def remove_tanh_symmetries(self, parameters_vectors, bias=True):
        result_parameters_vectors = np.array(parameters_vectors.copy())
        # Test/separate along the kernel parameters
        parameters_size = sum(tree_leaves(tree_map(lambda leaf: reduce(lambda a, b: a * b, leaf), self._parameters_shapes)))
        leaves, treedef = tree_flatten(self._parameters_shapes)
        
        # construct tree of parameters indices
        leaves_indices = []
        i = 0
        for leaf in leaves:
            size = reduce(lambda a, b: a * b, leaf)
            leaves_indices.append(np.arange(parameters_size)[i:i + size].reshape(leaf))
            i += size
        parameters_indices = tree_unflatten(treedef, leaves_indices)
        
        reflection_indices = []
        layer_keys = list(self._parameters_shapes["params"].keys())
        for i, layer_key in enumerate(layer_keys[:-1]):
            layer_indices = parameters_indices["params"][layer_keys[i]]
            following_layer_indices = parameters_indices["params"][layer_keys[i + 1]]
            bias_indices = layer_indices["bias"]
            kernel_indices = layer_indices["kernel"]
            following_kernel_indices = following_layer_indices["kernel"]
            if bias:
                reflection_indices += jnp.concatenate([bias_indices.reshape((-1, 1)), kernel_indices.T, following_kernel_indices], axis=-1).tolist()
            else:
                reflection_indices += jnp.concatenate([kernel_indices.T, bias_indices.reshape((-1, 1)), following_kernel_indices], axis=-1).tolist()
        
        # reflect parameters
        for indices in reflection_indices:
            selection = result_parameters_vectors[:, indices[0]] < 0.0
            selection = np.argwhere(selection)
            result_parameters_vectors[selection, indices] = -result_parameters_vectors[selection, indices]
        return result_parameters_vectors
    
    def activation_symmetries(self):
        parameters_size = sum(tree_leaves(tree_map(lambda leaf: reduce(lambda a, b: a * b, leaf), self._parameters_shapes)))
        layer_keys = list(self._parameters_shapes["params"].keys())
        result_transformations = []
        index = 0
        for i, layer_key in enumerate(layer_keys[:-1]):
            layer_shapes = self._parameters_shapes["params"][layer_keys[i]]
            following_layer_shapes = self._parameters_shapes["params"][layer_keys[i + 1]]
            bias_shape = layer_shapes["bias"]
            kernel_shape = layer_shapes["kernel"]
            following_bias_shape = following_layer_shapes["bias"]
            following_kernel_shape = following_layer_shapes["kernel"]
            symmetries = list(product([-1, 1], repeat=bias_shape[0]))
            
            for symmetry in symmetries:
                transformation_diagonal = np.ones(parameters_size)
                j = index
                
                bias = np.array(symmetry)
                old_j = j
                j += bias.flatten().shape[0]
                transformation_diagonal[old_j:j] = bias
                
                kernel = np.stack([symmetry] * kernel_shape[0], axis=0)
                old_j = j
                j += kernel.flatten().shape[0]
                transformation_diagonal[old_j:j] = kernel.flatten()
                
                following_kernel = np.stack([symmetry] * following_kernel_shape[1], axis=1)
                old_j = j + following_bias_shape[0]
                j += following_kernel.flatten().shape[0] + following_bias_shape[0]
                transformation_diagonal[old_j:j] = following_kernel.flatten()
                
                result_transformations.append(np.diag(transformation_diagonal))
            index += reduce(lambda a, b: a * b, bias_shape) + reduce(lambda a, b: a * b, kernel_shape)
            
        return np.array(result_transformations)
        
    def permutation_symmetries(self):
        parameters_size = sum(tree_leaves(tree_map(lambda leaf: reduce(lambda a, b: a * b, leaf), self._parameters_shapes)))
        leaves, treedef = tree_flatten(self._parameters_shapes)
        
        # construct tree of parameters indices
        leaves_indices = []
        i = 0
        for leaf in leaves:
            size = reduce(lambda a, b: a * b, leaf)
            leaves_indices.append(np.arange(parameters_size)[i:i + size].reshape(leaf))
            i += size
        parameters_indices = tree_unflatten(treedef, leaves_indices)
        
        # collect all transformations
        layer_keys = list(self._parameters_shapes["params"].keys())
        result_transformations = []
        index = 0
        for i, layer_key in enumerate(layer_keys[:-1]):
            layer_indices = parameters_indices["params"][layer_keys[i]]
            following_layer_indices = parameters_indices["params"][layer_keys[i + 1]]
            bias_indices = layer_indices["bias"]
            kernel_indices = layer_indices["kernel"]
            following_bias_indices = following_layer_indices["bias"]
            following_kernel_indices = following_layer_indices["kernel"]
            size = bias_indices.shape[0] + following_bias_indices.shape[0] + reduce(lambda a, b: a * b, kernel_indices.shape) + reduce(lambda a, b: a * b, following_kernel_indices.shape)
            
            permutation_matrix = np.eye(bias_indices.shape[0])
            permutation_matrices = np.array(list(permutations(permutation_matrix)))
            permutation_count = permutation_matrices.shape[0]
            
            bias_indices_permuted = permutation_matrices @ bias_indices
            kernel_indices_permuted = permutation_matrices @ kernel_indices.T
            following_bias_indices_permuted = np.stack([following_bias_indices] * permutation_count, axis=0)
            following_kernel_indices_permuted = permutation_matrices @ following_kernel_indices
            
            indices_stacked = np.concatenate([bias_indices_permuted, kernel_indices_permuted.reshape((permutation_count, -1)), following_bias_indices_permuted, following_kernel_indices_permuted.reshape((permutation_count, -1))], axis=-1)
            indices_permuted = np.stack([np.arange(parameters_size)] * permutation_count)
            indices_permuted[:, index:size] = indices_stacked
            index += size
            
            for element in indices_permuted:
                result_transformations.append(np.eye(element.shape[0])[element].T)
            
        return np.array(result_transformations)
