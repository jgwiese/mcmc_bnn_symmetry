import time
import jax
import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as distributions
import math
from tqdm import tqdm
from multiprocessing import Pool
import os


def run(rng_key, prob_model, data, num_warmup, num_samples_total, num_chains, progress_bar=False):
    # calcuations
    num_samples_chain = int(math.floor(1.0 * num_samples_total / num_chains))
    num_samples_rest = num_samples_total - num_chains * num_samples_chain
    assert num_samples_rest == 0, "num_samples_total should be divisible by num_chains"

    parallel_chains = jax.local_device_count()
    num_runs = int(jnp.floor(1.0 * num_chains / parallel_chains))
    num_runs_rest = num_chains - num_runs * parallel_chains
    assert num_runs_rest == 0, "num_chains should be divisible by jax.local_device_count()"
    
    # final samples
    samples = {}

    # inference loop
    kernel = numpyro.infer.NUTS(prob_model)
    mcmc = numpyro.infer.MCMC(
        kernel,
        num_warmup=num_warmup,
        num_samples=num_samples_chain * parallel_chains,
        num_chains=parallel_chains,
        progress_bar=progress_bar,
    )
    
    for run in tqdm(range(num_runs)):
        rng_key, rng_key_ = jax.random.split(rng_key)
        start = time.time()
        mcmc.run(rng_key_, data)
        end = time.time()
        #print(run, "run duration:", end - start)

        # concatenate samples
        run_samples = mcmc.get_samples()
        for key in run_samples.keys():
            if key not in samples:
                samples[key] = run_samples[key]
            else:
                samples[key] = jnp.concatenate([samples[key], run_samples[key]], axis=0)

    return samples

