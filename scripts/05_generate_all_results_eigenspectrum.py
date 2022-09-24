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
import matplotlib
import matplotlib.pyplot as plt


parser = argparse.ArgumentParser(
    description="run post hoc experiment with fixed identifiability constraints"
)
parser.add_argument("--verbose", type=bool, default=True)
parser.add_argument("--start", type=int, default=0)

def get_knn_sc_eigenvalues(samples, n=256):
    a = graphs.knn_graph(nodes=samples, k=10)
    d = graphs.degree_matrix(a)
    l = graphs.laplacian(a=a, d=d, normalized=True)
    eigenvalues, eigenvectors = graphs.spectrum(l=l, k=n, normalized=False)
    return eigenvalues

if __name__ == "__main__":
    args = parser.parse_args()
    rng_key = jax.random.PRNGKey(0)

    folder = global_settings.PATH_RESULTS
    files = [f for f in os.listdir(folder) if os.path.isfile(os.path.join(global_settings.PATH_RESULTS, f)) and f.split('.')[-1] == "gz"]
    print(files)
    
    all_results_str = "identifier spread_gt spread_fixed spread_custom rel_spread_fixed rel_spread_custom\n"
    for f in files[args.start:]:
        result = results.ResultSample.load_from_file(os.path.join(global_settings.PATH_RESULTS, f))
        experiment = experiments.FactoryExperiment(result.experiment_type, **{"settings": result.settings})()
        print(result.identifier, experiment._settings)
        base_path = os.path.join(global_settings.PATH_RESULTS, result.identifier)

        samples_gt = result.samples["parameters"]
        samples_fixed = jnp.load(os.path.join(base_path, f"{result.identifier}_fixed.npy"))
        samples_custom = jnp.load(os.path.join(base_path, f"{result.identifier}_2_1024_rbf.npy"))
        
        # eigenspectrum
        lambdas_gt = get_knn_sc_eigenvalues(standardize(samples_gt))
        lambdas_fixed = get_knn_sc_eigenvalues(standardize(samples_fixed))
        lambdas_custom = get_knn_sc_eigenvalues(standardize(samples_custom))

        # plotting
        amount_1 = 256
        amount_2 = 64
        lambdas_indices_1 = np.arange(len(lambdas_gt[:amount_1]))
        lambdas_indices_2 = np.arange(len(lambdas_gt[:amount_2]))
        s=9
        cm = matplotlib.cm.get_cmap("gist_rainbow")

        figure = plt.figure(figsize=(12, 3))
        ax1 = figure.add_subplot(1, 2, 1)
        ax1.set_xlabel(r"$i$")
        ax1.set_ylabel(r"$\lambda_i$")
        ax1.plot(lambdas_indices_1, lambdas_gt[:amount_1], label="unconstrained")
        ax1.plot(lambdas_indices_1, lambdas_fixed[:amount_1], label="constrained (fixed)")
        ax1.plot(lambdas_indices_1, lambdas_custom[:amount_1], label="constrained (custom)")

        ax2 = figure.add_subplot(1, 2, 2)
        ax2.set_xlabel(r"$i$")
        ax2.set_ylabel(r"$\lambda_i$")
        ax2.scatter(lambdas_indices_2, lambdas_gt[:amount_2], marker="o", s=s)
        ax2.plot(lambdas_indices_2, lambdas_gt[:amount_2])
        ax2.scatter(lambdas_indices_2, lambdas_fixed[:amount_2], marker="o", s=s)
        ax2.plot(lambdas_indices_2, lambdas_fixed[:amount_2])
        ax2.scatter(lambdas_indices_2, lambdas_custom[:amount_2], marker="o", s=s)
        ax2.plot(lambdas_indices_2, lambdas_custom[:amount_2])
        ax1.legend()

        figure.savefig(os.path.join(global_settings.PATH_FIGURES, "spectrum_{}.png".format(result.identifier)), bbox_inches="tight", dpi=192, transparent=True)
        
        # save results
        #base_path = os.path.join(global_settings.PATH_RESULTS, result.identifier)
        #if args.verbose:
        #    print(base_path)
       # if not os.path.exists(base_path):
       #     os.mkdir(base_path)
    print(all_results_str)
