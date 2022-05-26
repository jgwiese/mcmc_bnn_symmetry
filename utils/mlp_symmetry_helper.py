from functools import reduce
from itertools import product, permutations
import numpy as np
import scipy.sparse as sp
from jax.tree_util import tree_map, tree_leaves, tree_flatten, tree_unflatten
import jax.numpy as jnp
import utils
from tqdm import tqdm
from typing import Dict, List


def similarity_measure(node_0, node_1, sigma=1.0):
    coords_0, coords_1 = node_0[1], node_1[1]
    distance = utils.p_norm(coords_0 - coords_1, p=2.0)
    similarity = jnp.exp(- (1.0 / (2 * sigma**2)) * distance)
    return similarity 

def potential_positive(node_0, node_1):
    rv_0, rv_1 = node_0[2], node_1[2]
    similarity = similarity_measure(node_0, node_1)
    dissimilarity = 1.0 - similarity
    #similarity = 0.50001
    #dissimilarity = 1.0 - similarity
    #coords_0, coords_1 = node_0[1], node_1[1]
    #distance = utils.p_norm(coords_0 - coords_1, p=2.0)
    #similarity = 1.0 / distance
    #dissimilarity = distance
    if rv_0 == rv_1:    
        return similarity
    else:
        return dissimilarity


def potential_positive_nodes(node_0, node_1, sigma=1.0):
    distance = utils.p_norm(node_0.value["coordinates"] - node_1.value["coordinates"], p=2.0)
    similarity = jnp.exp(- (1.0 / (2 * sigma ** 2)) * distance)
    #similarity = 0.9
    if node_0.value["label"] == node_1.value["label"]:
        return similarity
    else:
        return 1.0 - similarity

class Node:
    def __init__(self, index: int, value: Dict):
        self.index = index
        self.value = value


# Structure
# I will have n parameters samples (MCMC). But each sample yields multiple nodes dependent on the amount of hidden neurons.
# I will save in values = {coordinates, label, sample_index}
# I will also have a dict for samples_dict = {sample_nodes = []}
# I will work with the nodes only and then in the last step convert back to parameters.


class MLPSymmetryHelper:
    # so far only works for single hidden layer MLPs
    def __init__(self, parameters_template, activation_function):
        self._parameters_shapes = tree_map(lambda x: jnp.array(x.shape), parameters_template)
        self._activation_function = activation_function
    
    def get_hidden_neuron_coordinates_PRE(self, samples_parameters):
        hidden_size = self._parameters_shapes["params"]["layers_0"]["bias"][0]
        hidden_neuron_coordinates = jnp.stack([
            samples_parameters[:, 0:hidden_size],
            samples_parameters[:, hidden_size:2 * hidden_size],
            samples_parameters[:, 2 * hidden_size + 1:]
        ], axis=-2)
        return hidden_neuron_coordinates
    
    def generate_node_dictionary_PRE(self, samples_parameters):
        hidden_neuron_coordinates = self.get_hidden_neuron_coordinates_PRE(
            samples_parameters=samples_parameters
        )
        n, dim, hidden_size = hidden_neuron_coordinates.shape
        
        result = {}
        for i, cp in enumerate(hidden_neuron_coordinates):
            for h, hidden_neuron in enumerate(cp.transpose(1, 0)):
                node_index = int(i * hidden_size  + h)
                node = Node(
                    index=node_index,
                    value={
                        "coordinates": hidden_neuron,
                        "label": h,
                        "sample_index": i
                    }
                )    
                result[node_index] = node
        return result 
    
    def generate_samples_nodes_dictionary_PRE(self, nodes):
        hidden_size = self._parameters_shapes["params"]["layer_0"]["bias"][0]
        n = int(len(nodes) / hidden_size)
        samples_nodes = {}
        
        for i in range(n):
            sample_nodes = []
            for h in range(hidden_size):
                index = int(i * hidden_size + h)
                node = nodes[index]
                assert node.index == index, "wrong node index"
                assert node.value["sample_index"] == i, "wrong sample index"
                sample_nodes.append(nodes[index])
            samples_nodes[i] = sample_nodes
        return samples_nodes
    
    def knn_graph(self, nodes: List[Node], k: int=1):
        nodes_coordinates_vector = jnp.stack([node.value["coordinates"] for node in nodes])
        adjacency_matrix = sp.lil_matrix((len(nodes), len(nodes)), dtype=jnp.float32)
        for i, node in enumerate(nodes):
            vs = nodes_coordinates_vector - node.value["coordinates"]
            distances = utils.p_norm(vs, p=2.0)
            nearest_neighbors_indices = jnp.argsort(distances)
            k_selection = nearest_neighbors_indices[1:1 + k]
            adjacency_matrix[i, k_selection] = 1.0
            adjacency_matrix[k_selection, i] = 1.0
        return adjacency_matrix.tocsr()
    
    def remove_permutation_symmetries_mrf_custom_2(self, samples_parameters, graph_samples_size, iterations=1):
        nodes = self.generate_node_dictionary_PRE(samples_parameters=samples_parameters)
        samples_nodes = self.generate_samples_nodes_dictionary_PRE(nodes=nodes)
        
        # init fully connected graph (TODO: this goes much cheaper than using knn...)
        print("\ninit graph")
        hidden_size = self._parameters_shapes["params"]["layers_0"]["bias"][0]
        a_size = int(hidden_size * graph_samples_size)
        assert len(nodes) >= a_size
        init_nodes = [nodes[key] for key in list(nodes.keys())[:a_size]]
        #a = self.knn_graph(nodes=init_nodes, k=len(init_nodes) - 1)
        a = self.knn_graph(nodes=init_nodes, k=3)
        
        # set negative edges    
        print("\nset negative edges")
        for i in range(graph_samples_size):
            sample_nodes = samples_nodes[i]
            indices = [node.index for node in sample_nodes]
            for j in indices:
                a[j, indices] = -1.0
            a[indices, indices] = 0.0
        
        # label initial graph
        print("\nlabel initial graph")
        for asdf in range(iterations):
            changes = 0
            permutation_symmetries = list(permutations(jnp.arange(hidden_size).tolist()))
            for i in range(0, len(samples_nodes)):
                #print("\nsample", i)
                sample_nodes = samples_nodes[i]
                labels = jnp.asarray([node.value["label"] for node in sample_nodes])
                log_probabilities = np.zeros(len(permutation_symmetries))
                # for this set of nodes calculate probability for each ps
                for k, ps in enumerate(permutation_symmetries):
                    labels_permuted = labels[jnp.asarray(ps)]
                    probability = 1.0
                    log_probability = 0.0
                    for j, node in enumerate(sample_nodes):
                        row = jnp.asarray(a[node.index].todense()).squeeze()
                        connected_nodes_indices = jnp.argwhere(row > 0.0).reshape((-1, ))
                        for cni in connected_nodes_indices.tolist():
                            cn = nodes[cni]
                            node_modified = Node(
                                index=node.index,
                                value = {
                                    "coordinates": node.value["coordinates"],
                                    "label": ps[j],
                                    "samle_index": node.value["sample_index"]
                                }
                            )
                            potential = potential_positive_nodes(node_0=node_modified, node_1=cn)
                            #print("node_0(", node_modified.index, node_modified.value["label"], "), node_1(", cn.index, cn.value["label"], "), potential:", potential)
                            log_probability += jnp.log(potential)
                            #print(labels_permuted, probability, log_probability)
                        #print("next node")
                    log_probabilities[k] = log_probability
                    #print("\nnew permutation")
                #print("sample", i, probabilities, np.argmax(probabilities))
                #print("sample", i, log_probabilities, np.argmax(log_probabilities))
                # change labeling of nodes now
                ps_best = permutation_symmetries[np.argmax(log_probabilities)]
                for j, node in enumerate(sample_nodes):
                    if node.value["label"] != ps_best[j]:
                        changes += 1
                    node.value["label"] = ps_best[j]
            print("iteration", asdf, "changing a node:", changes)
                
        return a, init_nodes
        
    
    def remove_permutation_symmetries_mrf_custom(self, parameters_vector, init_size=32, matrix_size=64):
        # Create combined posterior of permutation symmetry relevant parameters
        hidden = self._parameters_shapes["params"]["layers_0"]["bias"][0]
        combined_parameters = jnp.stack([
            parameters_vector[:, 0:hidden],
            parameters_vector[:, hidden:2 * hidden],
            parameters_vector[:, 2 * hidden + 1:]
        ], axis=-2)
        n, dim, hidden = combined_parameters.shape
        node_coords = combined_parameters.transpose(0, 2, 1).reshape((-1, dim))
        
        # result vectors
        combined_parameters_permuted = [combined_parameters[0]]
        
        # Now process one parameters vector after another, build graph, calculate energy for each permutation in graph.
        # node = i, edge = (i, j, value)
        nodes = (0 * hidden +  jnp.arange(hidden)).tolist()
        node_values = jnp.arange(hidden).tolist() # can only be in {0, ..., hidden}
        edges = []
        a_current = sp.lil_matrix((len(nodes), len(nodes)), dtype=jnp.float32)
        a_new = sp.lil_matrix((len(nodes), len(nodes)), dtype=jnp.float32)
        a_current[:hidden, :hidden] = -1.0
        a_current[jnp.arange(hidden), jnp.arange(hidden)] = 0.0
        for i in tqdm(range(1, combined_parameters.shape[0])):
            cp = combined_parameters[i]
            node_coords_current = node_coords[jnp.array(nodes)]
            
            nodes += (i * hidden +  jnp.arange(hidden)).tolist()
            a_last = a_current.copy()
            a_current = sp.lil_matrix((len(nodes), len(nodes)), dtype=jnp.float32)
            a_current[:a_last.shape[0], :a_last.shape[1]] = a_last

            # by growing I will have a fully connected graph for sure and this is what I need.
            k = 1
            for h in range(hidden):
                # connect k closest nodes
                distances = utils.p_norm(node_coords_current - cp[:, h], p=2.0)
                nearest_neighbors_indices = jnp.argsort(distances)
                k_selection = nearest_neighbors_indices[:k]
                a_current[i * hidden + h, k_selection] = 1.0
                a_current[k_selection, i * hidden + h] = 1.0
                
                # connect the hidden nodes with each other
                a_current[i * hidden + h, jnp.arange(hidden) + i * hidden] = -1.0
                a_current[i * hidden + jnp.arange(hidden), i * hidden + jnp.arange(hidden)] = 0.0
            
            # calculate energies for all permutations and only allow one.
            permutation_symmetries = jnp.array(list(permutations(jnp.arange(hidden))))
            permutation_matrix = 0
            # now i will evaluate the probability for each permutation of the tuple of hidden nodes and choose the permutation of highest probability.
            probabilities = np.zeros(permutation_symmetries.shape[0])
            for k, ps in enumerate(permutation_symmetries):
                # I work with cliques of size 2
                hidden_node_values = ps
                ps_probability = 1.0
                for h in range(hidden):
                    node_index = i * hidden + h
                    for j in range(a_current[node_index].shape[1]):
                        # work on positive edges
                        if a_current[node_index, j] > 0.0:
                            # TODO, General a node: (index, coords, random variable value)
                            node_current = (node_index, cp[:, h], hidden_node_values[h])
                            node_reference = (j, node_coords[j], node_values[j])
                            potential = potential_positive(node_current, node_reference)
                            ps_probability *= potential
                probabilities[k] = ps_probability
            probabilities /= probabilities.sum()
            #print(probabilities, jnp.argmax(probabilities))
            
            # permute sample
            ps = permutation_symmetries[jnp.argmax(probabilities)]
            t = jnp.eye(hidden)[ps]
            combined_parameters_permuted.append(cp @ t)
            node_values += ps.tolist()
        combined_parameters_permuted = jnp.asarray(combined_parameters_permuted)
        parameters_vector_permuted = jnp.concatenate([
            combined_parameters_permuted[:, 0],
            combined_parameters_permuted[:, 1],
            parameters_vector[:, 2 * hidden:2 * hidden + 1],
            combined_parameters_permuted[:, 2]
        ], axis=-1)
        #print(parameters_vector_permuted.shape)

        return parameters_vector_permuted, a_current
    
    def remove_permutation_symmetries_amor(self, parameters_vector):
        """ TODO: Do for Mlp not only for single layer. """
        # Create combined posterior of permutation symmetry relevant parameters
        hidden = self._parameters_shapes["params"]["dense_0"]["bias"][0]
        combined = jnp.stack([
            parameters_vector[:, 0:hidden],
            parameters_vector[:, hidden:2 * hidden],
            parameters_vector[:, 2 * hidden + 1:]
        ], axis=-2)
        n, dim, hidden = combined.shape
        
        # Permute corresponding sets of parameters to have lowest distance to each other
        print("find clostest distance samples")
        permutation_symmetries = jnp.array(list(permutations(jnp.arange(hidden))))
        permutation_matrices = np.zeros((n, ), dtype=np.int32)
        distances = np.inf * np.ones((n, ), dtype=np.float32)
        v0 = combined.reshape((-1, dim * hidden))[0]
        for i, ps in enumerate(permutation_symmetries):
            t = jnp.eye(hidden)[ps]
            combined_transformed = combined @ t
            differences = combined_transformed.reshape((-1, dim * hidden)) - v0
            distances_transformed = np.asarray(utils.p_norm(differences, p=2.0))
            selection = np.asarray(jnp.argwhere(distances_transformed < distances)).squeeze()
            distances[selection] = distances_transformed[selection]
            permutation_matrices[selection] = i
        
        # permute samples
        print("permute samples")
        combined_permuted = []
        for i, v in enumerate(combined):
            ps = permutation_symmetries[permutation_matrices[i]]
            t = jnp.eye(hidden)[ps]
            combined_permuted.append(v @ t)
        combined_permuted = jnp.asarray(combined_permuted)
        print(combined_permuted.shape)
        parameters_vector_permuted = jnp.concatenate([
            combined_permuted[:, 0],
            combined_permuted[:, 1],
            parameters_vector[:, 2 * hidden:2 * hidden + 1],
            combined_permuted[:, 2]
        ], axis=-1)
        print(parameters_vector_permuted.shape)
        return parameters_vector_permuted
        
        """
        full_permutation_symmetries = self.permutation_symmetries()
        parameters_vector_permuted = []
        for i, v in enumerate(parameters_vector):
            # TODO: Attention: Assumes that permutation_symmetries and full_permutation_symmetries have the same order... test.
            t = full_permutation_symmetries[permutation_matrices[i]]
            parameters_vector_permuted.append(v @ t)
        parameters_vector_permuted = jnp.asarray(parameters_vector_permuted)
        print("shape", parameters_vector_permuted.shape)
        return parameters_vector_permuted
        """
    
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

    def remove_tanh_symmetries_svm(self, parameters_vectors):
        """ TODO: Make this method applicable in general. Requires per layer plane and mergin of parameters. """
        pass
    
    def remove_permutation_symmetries_manual(self, identifiability_constraints):
        """ TODO: Method to remove symmetries by provided constraints. """
        pass
    
    def remove_permutation_symmetries_spectral_clustering(self):
        """ TODO """
        pass
    
    def remove_permutation_symmetries_mrf(self):
        """ TODO """
        pass
    
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
