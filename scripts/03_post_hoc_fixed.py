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
    description="run post hoc experiment with fixed identifiability constraints"
)
parser.add_argument("--verbose", type=bool, default=True)


if __name__ == "__main__":
    args = parser.parse_args()
    rng_key = jax.random.PRNGKey(0)

    folder = global_settings.PATH_RESULTS
    files = [f for f in os.listdir(folder) if os.path.isfile(os.path.join(global_settings.PATH_RESULTS, f)) and f.split('.')[-1] == "gz"]
    print(len(files))
    
    for f in files:
        result = results.ResultSample.load_from_file(os.path.join(global_settings.PATH_RESULTS, f))
        experiment = experiments.FactoryExperiment(result.experiment_type, **{"settings": result.settings})()
        if args.verbose:
            print(experiment._settings)
        sequential_helper = equioutput.SequentialHelper(
            transformation=experiment._model_transformation,
            dataset=experiment._dataset
        )

        sh_fixed = equioutput.SymmetryHelperFixed(
            parameters_template=experiment._model_transformation.init(rng_key, experiment._dataset[0][0]),
            activation_function="tanh"
        )

        samples_parameters_mirrored = result.samples["parameters"]
        samples_parameters_mirrored = sh_fixed.remove_tanh_symmetries(samples_parameters_mirrored, bias=True)
        samples_parameters_mirrored = sh_fixed.remove_permutation_symmetries(samples_parameters_mirrored, bias=True)
        
        # save results
        base_path = os.path.join(global_settings.PATH_RESULTS, result.identifier)
        if args.verbose:
            print(base_path)
        if not os.path.exists(base_path):
            os.mkdir(base_path)
        
        jnp.save(os.path.join(base_path, f"{result.identifier}_fixed.npy"), samples_parameters_mirrored)
