import argparse
import jax
import jax.numpy as jnp
import flax.linen as nn
import numpyro
from multiprocessing import Pool
from tqdm import tqdm
import json
import datetime
import os
from dataclasses import dataclass
from data import datasets
import models
import transformation
import utils


parser = argparse.ArgumentParser(
    description="run experiment"
)
parser.add_argument("--output_path", type=str, default=".")
parser.add_argument("--dataset", type=str, default="izmailov", help="one of: sinusoidal, izmailov")
parser.add_argument("--dataset_normalization", type=str, default="standardization")
parser.add_argument("--hidden_layers", type=int, default=1)
parser.add_argument("--hidden_neurons", type=int, default=1)
parser.add_argument("--activation", type=str, default="tanh")
parser.add_argument("--activation_last_layer", type=str, default="none")
parser.add_argument("--num_warmup", type=int, default=1)
parser.add_argument("--statistic", type=str, default="full", help="full / reduced")
parser.add_argument("--statistic_p", type=float, default="0.95")
parser.add_argument("--samples_per_mode", type=int, default=2**6)
parser.add_argument("--identifiable_modes", type=int, default=1)
parser.add_argument("--pool_size", type=int, default=1)
parser.add_argument("--seed", type=int, default=0)


if __name__ == "__main__":
    args = parser.parse_args()
    settings = utils.SettingsExperiment(**vars(args))
    
    rng_key = jax.random.PRNGKey(args.seed)
    experiment_id = "{}_hlayers{}_hneurons{}_{}".format(args.dataset, args.hidden_layers, args.hidden_neurons, args.statistic)
    date = datetime.datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
    output_path = os.path.join(os.path.join(os.getcwd(), args.output_path), experiment_id)
    if not os.path.exists(output_path):
        os.mkdir(output_path)
    print("output path", output_path)
    
    dataset = datasets.create_dataset(settings)
    model_transformation = transformation.create_model_transformation(settings, dataset)
    model = models.create_model(settings, dataset, model_transformation)
    
    # sampler
    ## statistics
    rng_key, rng_key_ = jax.random.split(rng_key)
    sh = utils.MLPSymmetryHelper(
        parameters_template=model_transformation.init(rng_key_, dataset[0][0]),
        activation_function="tanh"
    )
    modes = args.identifiable_modes
    if args.statistic == "full":
        modes *= sh.symmetries_size()
    num_chains = int(2**jnp.ceil(jnp.log2(utils.bounded_expected_number_of_chains(n=modes, p=args.statistic_p))))
    num_samples = int(max(2**jnp.ceil(jnp.log2(modes * args.samples_per_mode)), num_chains))
    num_samples_per_chain = int(num_samples / num_chains)
    print("parameters_size: {}, symmetries: {}, num_chains: {}, num_samples_per_chain: {}, num_samples: {}".format(model_transformation.parameters_size(dataset[0][0]), sh.symmetries_size(), num_chains, num_samples_per_chain, num_samples))
    
    ## sampler
    global_kernel = numpyro.infer.NUTS(model)
    global_mcmc = numpyro.infer.MCMC(
        global_kernel,
        num_warmup=args.num_warmup,
        num_samples=num_samples_per_chain,
        num_chains=1,
        progress_bar=False
    )
    
    def infer_parallel(key):
        global_mcmc.run(key)
        return global_mcmc.get_samples()

    rng_key, *sub_keys = jax.random.split(rng_key, 1 + num_chains)
    with Pool(args.pool_size) as p:
        samples_parallel = list(tqdm(p.imap(infer_parallel, sub_keys), total=num_chains))
    
    samples = {}
    for samples_run in samples_parallel:
        for key in samples_run.keys():
            if key not in samples:
                samples[key] = samples_run[key]
            else:
                samples[key] = jnp.concatenate([samples[key], samples_run[key]])
    
    # results
    dataset_file_name = "{}_dataset.npy".format(date)
    jnp.save(os.path.join(output_path, dataset_file_name), dataset.data)
    
    samples_file_names = {}
    for key in samples.keys():
        samples_file_name = "{}_samples_{}.npy".format(date, key)
        jnp.save(os.path.join(output_path, samples_file_name), samples[key])
        samples_file_names[key] = samples_file_name
    
    results = {
        "date": date,
        "settings": vars(args),
        "dataset_file_name": dataset_file_name,
        "samples_file_names": samples_file_names
    }
    
    with open(os.path.join(output_path, "{}_results.json".format(date)), "w") as f:
        json.dump(results, f)
    
