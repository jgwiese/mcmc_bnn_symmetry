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
    print(files)
    
    all_results_str = "identifier spread_gt spread_fixed spread_custom rel_spread_fixed rel_spread_custom\n"
    for f in files:
        result = results.ResultSample.load_from_file(os.path.join(global_settings.PATH_RESULTS, f))
        experiment = experiments.FactoryExperiment(result.experiment_type, **{"settings": result.settings})()
        print(result.identifier, experiment._settings)
        base_path = os.path.join(global_settings.PATH_RESULTS, result.identifier)

        samples_gt = result.samples["parameters"]
        samples_fixed = jnp.load(os.path.join(base_path, f"{result.identifier}_fixed.npy"))
        samples_custom = jnp.load(os.path.join(base_path, f"{result.identifier}_2_1024_rbf.npy"))
        # spread
        spread_gt = evaluation.spread(samples_gt)
        spread_fixed = evaluation.spread(samples_fixed)
        spread_custom = evaluation.spread(samples_custom)
        rel_spread_fixed = 1.0 - spread_fixed / spread_gt
        rel_spread_custom = 1.0 - spread_custom / spread_gt
        result_str = "{} {} {} {} {} {}\n".format(result.identifier, spread_gt, spread_fixed, spread_custom, rel_spread_fixed, rel_spread_custom)
        all_results_str += result_str
        
        # save results
        #base_path = os.path.join(global_settings.PATH_RESULTS, result.identifier)
        #if args.verbose:
        #    print(base_path)
       # if not os.path.exists(base_path):
       #     os.mkdir(base_path)
    print(all_results_str)
