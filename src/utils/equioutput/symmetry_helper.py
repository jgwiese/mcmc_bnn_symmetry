import jax
import jax.numpy as jnp
import numpy as np
from jax.tree_util import tree_map, tree_flatten, tree_unflatten
from typing import Dict, Any, List
from utils import equioutput, graphs


class NeuronParametersIndices:
    def __init__(self, parameters_indices):
        self._parameters_indices = parameters_indices
    
    def __repr__(self):
        return "NeuronParametersIndices:" + self._parameters_indices.__repr__()
    
    @property
    def parameters_indices(self):
        return self._parameters_indices


class LayerParametersIndices:
    def __init__(self, neurons_parameters_indices: Dict[int, NeuronParametersIndices]):
        self._neurons_parameters_indices = neurons_parameters_indices
    
    def __repr__(self):
        return self._neurons_parameters_indices.__repr__()
    
    @property
    def neurons_parameters_indices(self):
        return self._neurons_parameters_indices


class StructuredSequentialSamplesParameters:
    def __init__(self, samples_parameters, layers_parameters_indices: Dict[int, LayerParametersIndices]):
        assert len(samples_parameters.shape) == 2
        self._samples_parameters = np.array(samples_parameters)
        self._layers_parameters_indices = layers_parameters_indices
    
    def __repr__(self):
        repr_str = self._parameters.__repr__()
        return repr_str + self._layer_parameters_indices.__repr__()
    
    @property
    def samples_parameters(self):
        return self._samples_parameters
    
    @property
    def layers_parameters_indices(self):
        return self._layers_parameters_indices


class SequentialHelper:
    def __init__(self, transformation, dataset):
        self._transformation = transformation
        self._dataset = dataset
        self._parameters_shapes = tree_map(lambda x: jnp.array(x.shape), self._transformation.init(jax.random.PRNGKey(0), self._dataset[0][0]))
    
    def _parameters_indices(self):
        indices = jnp.arange(self._transformation.parameters_size(self._dataset[0][0]))
        leaves, treedef = tree_flatten(self._parameters_shapes)
        leaves_indices = []
        i = 0
        for leaf in leaves:
            size = jnp.prod(leaf)
            leaves_indices.append(indices[i:i+size].reshape(leaf))
            i += size
        parameters_indices = tree_unflatten(treedef, leaves_indices)
        return parameters_indices

    def structured_sequential_samples_parameters(self, samples_parameters):
        layers_parameters = {}
        parameters_indices = self._parameters_indices()
        layer_keys = list(parameters_indices["params"].keys())
        for l, layer_key in enumerate(layer_keys[:-1]):
            bias_indices = parameters_indices["params"][layer_key]["bias"]
            kernel_indices = parameters_indices["params"][layer_key]["kernel"]
            fkernel_indices = parameters_indices["params"][layer_keys[l + 1]]["kernel"]
            
            layer_neurons_parameters_tensor = jnp.concatenate([
                bias_indices[jnp.newaxis],
                kernel_indices,
                fkernel_indices.T
            ], axis=0).T

            layer_neurons_parameters = {}
            for h, lnpt in enumerate(layer_neurons_parameters_tensor):
                layer_neurons_parameters[h] = NeuronParametersIndices(lnpt)
            layers_parameters[l] = LayerParametersIndices(layer_neurons_parameters)
        structured_sequential_samples_parameters = StructuredSequentialSamplesParameters(samples_parameters, layers_parameters)
        return structured_sequential_samples_parameters
        


class SymmetryHelper: # TODO: SymmetryRemoverCustom?
    def __init__(self, structured_sequential_samples_parameters):
        self._structured_sequential_samples_parameters = structured_sequential_samples_parameters
        self._number_of_samples = len(self._structured_sequential_samples_parameters.samples_parameters)
        self._number_of_layers = len(self._structured_sequential_samples_parameters.layers_parameters_indices)
        self._similarity_matrix = None

    def hidden_layer_subspace(self, layer: int):
        assert layer < self._number_of_layers
        
        # layer indices structure
        layer_parameters_indices = self._structured_sequential_samples_parameters.layers_parameters_indices[layer]

        # construct theta hidden parameters subspace and find separating hyperplane in it
        subspace = []
        for h in layer_parameters_indices.neurons_parameters_indices.keys():
            neuron_indices = layer_parameters_indices.neurons_parameters_indices[h]
            parameters_h = self._structured_sequential_samples_parameters.samples_parameters[:, neuron_indices.parameters_indices]
            subspace.append(parameters_h)
        subspace = jnp.stack(subspace, axis=1)
        return subspace
    
    def remove_tanh_symmetries(self, layer: int):
        layer_parameters_indices = self._structured_sequential_samples_parameters.layers_parameters_indices[layer]
        subspace = self.hidden_layer_subspace(layer)

        # optimize hyperplane
        svm = equioutput.UnsupervisedSVMBinary(subspace.reshape(-1, subspace.shape[-1]))
        svm.optimize(2**5, 2**4, lr=0.1, report_at=1)

        # flip neurons
        for h in layer_parameters_indices.neurons_parameters_indices.keys():
            neuron_indices = layer_parameters_indices.neurons_parameters_indices[h]
            parameters_h = self._structured_sequential_samples_parameters.samples_parameters[:, neuron_indices.parameters_indices]
            selection_behind_hyperplane = parameters_h @ svm.normal < 0
            parameters_h[selection_behind_hyperplane] = -parameters_h[selection_behind_hyperplane]
            self._structured_sequential_samples_parameters.samples_parameters[:, neuron_indices.parameters_indices] = parameters_h

    def remove_permutation_symmetries(self, layer: int, iterations: int):
        # TODO: Use more the structures from above!
        layer_parameters_indices  = self._structured_sequential_samples_parameters.layers_parameters_indices[layer]
        subspace = self.hidden_layer_subspace(layer)
        n, hidden, dim = subspace.shape

        # similarity matrix
        if self._similarity_matrix is None:
            self._similarity_matrix = graphs.distance_knn_graph_dense(
                nodes=subspace.reshape((-1, dim)),
                k=hidden * 3
            )
            sele = self._similarity_matrix > 0
            self._similarity_matrix[sele] = np.power(self._similarity_matrix[sele], -1)
        #self._similarity_matrix = graphs.knn_graph_dense(nodes=subspace.reshape((-1, dim)), k=n * hidden)

        # iterative greedy knn
        current_labels = np.stack([np.arange(hidden)] * n, axis=0)
        rng_key = jax.random.PRNGKey(0)
        for i in range(iterations):
            ig_knn = equioutput.IterativeGreedyKNN()
            
            new_labels, indices = ig_knn.run(
                samples_subspace=subspace,
                labels=current_labels,
                similarity_matrix=self._similarity_matrix,
                indices=None
            )

            relabelings_total = jnp.logical_not(new_labels == current_labels).sum()
            print(i, relabelings_total)
            rng_key, rng_key_ = jax.random.split(rng_key)
            indices_subset = jax.random.permutation(rng_key_, jnp.arange(len(current_labels)))[:int(0.5 * len(current_labels))]
            current_labels[indices_subset] = new_labels[indices_subset]
        
        old_indices = []
        for h in layer_parameters_indices.neurons_parameters_indices.keys():
            old_indices.append(layer_parameters_indices.neurons_parameters_indices[h].parameters_indices)
        old_indices = jnp.stack(old_indices, 0)
        
        for i, labels in enumerate(current_labels):
            self._structured_sequential_samples_parameters.samples_parameters[i, old_indices.flatten()] = self._structured_sequential_samples_parameters.samples_parameters[i, old_indices[jnp.argsort(labels)].flatten()]
            #print(labels, old_indices[labels].flatten())

            

