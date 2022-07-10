import argparse
import global_settings
from utils import settings, experiments, results
import jax
from multiprocessing import Pool
from tqdm import tqdm
import jax.numpy as jnp
import numpy as np
import os
from data import standardize
from utils import results, experiments, settings, equioutput, evaluation, graphs


parser = argparse.ArgumentParser(
    description="run post hoc experiment"
)
parser.add_argument("--result_path", type=str)
parser.add_argument("--tanh_planes", type=int, default=2)
parser.add_argument("--k", type=int, default=1024)
parser.add_argument("--iterations", type=int, default=256)
parser.add_argument("--verbose", type=bool, default=True)
parser.add_argument("--pool_size", type=int, default=1)
parser.add_argument("--similarity_matrix", type=str, default="rbf")

def hidden_layer_subspace(structured_sequential_samples_parameters, layer):
    number_of_layers = len(structured_sequential_samples_parameters.layers_parameters_indices)
    assert layer < number_of_layers
    
    # layer indices structure
    layer_parameters_indices = structured_sequential_samples_parameters.layers_parameters_indices[layer]

    # construct theta hidden parameters subspace and find separating hyperplane in it
    subspace = []
    for h in layer_parameters_indices.neurons_parameters_indices.keys():
        neuron_indices = layer_parameters_indices.neurons_parameters_indices[h]
        parameters_h = structured_sequential_samples_parameters.samples_parameters[:, neuron_indices.parameters_indices]
        subspace.append(parameters_h)
    subspace = jnp.stack(subspace, axis=1)
    return subspace


if __name__ == "__main__":
    args = parser.parse_args()
    rng_key = jax.random.PRNGKey(0)
    
    result = results.ResultSample.load_from_file(args.result_path)
    experiment = experiments.FactoryExperiment(result.experiment_type, **{"settings": result.settings})()
    print(experiment._settings)
    sequential_helper = equioutput.SequentialHelper(
        transformation=experiment._model_transformation,
        dataset=experiment._dataset
    )

    structured_sequential_samples_parameters = sequential_helper.structured_sequential_samples_parameters(result.samples["parameters"])
    number_of_layers = len(structured_sequential_samples_parameters.layers_parameters_indices)
    for l in range(number_of_layers):
        layer = number_of_layers - l - 1
        layer_parameters_indices  = structured_sequential_samples_parameters.layers_parameters_indices[layer]
        
        # get relevant subspace
        subspace = hidden_layer_subspace(structured_sequential_samples_parameters, layer)
        subspace = standardize(subspace)
        
        # remove tanh symmetries
        ## optimize hyperplanes
        svms = []
        loss_values = []
        for i in range(args.tanh_planes):
            rng_key, rng_key_ = jax.random.split(rng_key, 2)
            svm = equioutput.UnsupervisedSVMBinary(subspace.reshape(-1, subspace.shape[-1]), rng_key=rng_key)
            loss_value = svm.optimize(epochs=2**5, batch_size=2**4, lr=0.1, report_at=1, verbose=args.verbose)
            #loss_value = svm.optimize(epochs=2**5, batch_size=2**8, lr=0.1, report_at=1, verbose=args.verbose)
            svms.append(svm)
            loss_values.append(loss_value)
        
        ## select best performing hyperplane.
        i = jnp.argmin(jnp.array(loss_values))
        svm = svms[i]
        print("best hyperplane", i)

        ## remove tanh symmetries
        for h in layer_parameters_indices.neurons_parameters_indices.keys():
            neuron_indices = layer_parameters_indices.neurons_parameters_indices[h]
            parameters_h = structured_sequential_samples_parameters.samples_parameters[:, neuron_indices.parameters_indices]
            selection_behind_hyperplane = parameters_h @ svm.normal < 0
            parameters_h[selection_behind_hyperplane] = -parameters_h[selection_behind_hyperplane]
            structured_sequential_samples_parameters.samples_parameters[:, neuron_indices.parameters_indices] = parameters_h

        # remove permutation symmetries
        subspace = hidden_layer_subspace(structured_sequential_samples_parameters, layer) # udpate necessary!
        subspace = standardize(subspace)
        n, hidden, dim = subspace.shape

        ## complicated way of determining indices to process for each core in parallel mode.
        all_indices = np.arange(n)
        core_indices = int(len(all_indices) * 1.0 / args.pool_size) + 1
        indices_parallel = []
        for i in range(args.pool_size - 1):
            indices_parallel.append(all_indices[i * core_indices: (i+1) * core_indices])
        indices_parallel.append(all_indices[(args.pool_size - 1) * core_indices:])

        ## similarity matrix
        dense = n * hidden < 20000
        if args.similarity_matrix == "rbf":
            if dense:
                print("using dense matrix")
                similarity_matrix = graphs.knn_graph_dense(nodes=subspace.reshape((-1, dim)), k=args.k)
            else:
                print("using sparse matrix")
                similarity_matrix = graphs.knn_graph(nodes=subspace.reshape((-1, dim)), k=args.k)
        else:
            if dense:
                print("using dense matrix")
                similarity_matrix = graphs.distance_knn_graph_dense(nodes=subspace.reshape((-1, dim)), k=args.k)
                sele = similarity_matrix > 0
                similarity_matrix[sele] = np.power(similarity_matrix[sele], -1)
            else:
                print("using sparse matrix")
                similarity_matrix = graphs.distance_knn_graph(nodes=subspace.reshape((-1, dim)), k=args.k).power(-1)

        ## iterative greedy knn
        current_labels = np.stack([np.arange(hidden)] * n, axis=0)
        convergence_counter = 0
        for i in range(args.iterations):
            ig_knn = equioutput.IterativeGreedyKNN()
            
            def knn_parallel(indices):
                return ig_knn.run(
                    samples_subspace=subspace,
                    labels=current_labels,
                    similarity_matrix=similarity_matrix,
                    indices=indices,
                    dense=dense
                )
            
            with Pool(args.pool_size) as p:
                parallel_results = list(p.imap(knn_parallel, indices_parallel))
            
            new_labels = np.zeros_like(current_labels)
            for element in parallel_results:
                new_labels_par, indices_par = element
                new_labels[indices_par] = new_labels_par
            """
            new_labels, indices = ig_knn.run(
                    samples_subspace=subspace,
                    labels=current_labels,
                    similarity_matrix=similarity_matrix,
                    indices=None
                )
            """
            ### early stopping
            relabelings_total = jnp.logical_not(new_labels == current_labels).sum()
            if relabelings_total == 0:
                convergence_counter += 1
            else:
                convergence_counter = 0
            if convergence_counter >= 8:
                break

            if args.verbose:
                print(i, relabelings_total)
            
            ### update only a fraction of new found labels (0.5)
            rng_key, rng_key_ = jax.random.split(rng_key)
            indices_subset = jax.random.permutation(rng_key_, jnp.arange(len(current_labels)))[:int(0.5 * len(current_labels))]
            current_labels[indices_subset] = new_labels[indices_subset]
        
        ## permute parameters
        old_indices = []
        for h in layer_parameters_indices.neurons_parameters_indices.keys():
            old_indices.append(layer_parameters_indices.neurons_parameters_indices[h].parameters_indices)
        old_indices = jnp.stack(old_indices, 0)
        
        for i, labels in enumerate(current_labels):
            structured_sequential_samples_parameters.samples_parameters[i, old_indices.flatten()] = structured_sequential_samples_parameters.samples_parameters[i, old_indices[jnp.argsort(labels)].flatten()]
            #print(labels, old_indices[labels].flatten())
    
    # save results
    base_path = '/'.join(args.result_path.split('/')[:-1] + [result.identifier])
    
    if not os.path.exists(base_path):
        os.mkdir(base_path)
    
    jnp.save(os.path.join(base_path, f"{result.identifier}_{args.tanh_planes}_{args.k}_{args.similarity_matrix}.npy"), structured_sequential_samples_parameters.samples_parameters)
