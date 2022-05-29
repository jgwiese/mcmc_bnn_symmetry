import argparse
import global_settings
from utils import settings, experiments, results
import jax
from multiprocessing import Pool
from tqdm import tqdm
import jax.numpy as jnp
import os


parser = argparse.ArgumentParser(
    description="run experiment"
)
parser.add_argument("--output_path", type=str, default=global_settings.PATH_RESULTS)
parser.add_argument("--dataset", type=str, default="izmailov", help="one of: sinusoidal, izmailov")
parser.add_argument("--dataset_normalization", type=str, default="standardization")
parser.add_argument("--hidden_layers", type=int, default=1)
parser.add_argument("--hidden_neurons", type=int, default=1)
parser.add_argument("--activation", type=str, default="tanh")
parser.add_argument("--activation_last_layer", type=str, default="none")
parser.add_argument("--num_warmup", type=int, default=1)
parser.add_argument("--statistic", type=str, default="reduced", help="full / reduced")
parser.add_argument("--statistic_p", type=float, default="0.95")
parser.add_argument("--samples_per_mode", type=int, default=2**6)
parser.add_argument("--identifiable_modes", type=int, default=1)
parser.add_argument("--pool_size", type=int, default=1)
parser.add_argument("--seed", type=int, default=0)


if __name__ == "__main__":
    args = parser.parse_args()
    experiment = experiments.ExperimentSample(
        settings=settings.SettingsExperimentSample(**vars(args))
    )
    
    def run_parallel(key):
        experiment.sample(key)
        return experiment._mcmc.get_samples()

    rng_key, *sub_keys = jax.random.split(jax.random.PRNGKey(experiment._settings.seed), experiment._sample_statistics["num_chains"] + 1)
    with Pool(experiment._settings.pool_size) as p:
        samples_parallel = list(tqdm(p.imap(run_parallel, sub_keys), total=experiment._sample_statistics["num_chains"]))

    samples = {}
    for samples_run in samples_parallel:
        for key in samples_run.keys():
            if key not in samples:
                samples[key] = samples_run[key]
            else:
                samples[key] = jnp.concatenate([samples[key], samples_run[key]])
    experiment._samples = samples

    file_name = experiment.save()
