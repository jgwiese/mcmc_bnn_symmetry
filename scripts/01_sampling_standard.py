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
parser.add_argument("--output_path", type=str, default=global_settings.PATH_PAPER_RESULTS)
parser.add_argument("--dataset", type=str, default="sinusoidal", help="one of: sinusoidal, izmailov, regression2d")
parser.add_argument("--dataset_normalization", type=str, default="standardization")
parser.add_argument("--hidden_layers", type=int, default=1)
parser.add_argument("--hidden_neurons", type=int, default=3)
parser.add_argument("--activation", type=str, default="tanh")
parser.add_argument("--activation_last_layer", type=str, default="none")
parser.add_argument("--num_warmup", type=int, default=1024)
parser.add_argument("--statistic", type=str, default="reduced", help="full / reduced")
parser.add_argument("--statistic_p", type=float, default=0.99)
parser.add_argument("--samples_per_chain", type=int, default=1)
parser.add_argument("--identifiable_modes", type=int, default=3)
parser.add_argument("--pool_size", type=int, default=1)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--overwrite_chains", type=int, default=None)


if __name__ == "__main__":
    args = parser.parse_args()
    experiment = experiments.ExperimentSampleStandard(
        settings=settings.SettingsExperimentSample(**vars(args))
    )
    
    if args.pool_size == 1:
        experiment.run()
    else:
        print("model transformation parameters {}".format(experiment._model_transformation.parameters_size(experiment._dataset[0][0])))

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

    print(f"saved as {experiment.save()}")
