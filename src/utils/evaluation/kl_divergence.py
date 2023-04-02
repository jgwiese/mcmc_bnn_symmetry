import jax.numpy as jnp
import utils.experiments as experiments


def kl_divergence(p_values, q_values):
    assert p_values.shape == q_values.shape
    
    # normalize
    p_values /= p_values.sum(1)[:, jnp.newaxis]
    q_values /= q_values.sum(1)[:, jnp.newaxis]
    
    return (p_values * (jnp.log(p_values + 1e-6) - jnp.log(q_values + 1e-6))).sum(-1)


def kl_divergence_grid(experiment:experiments.ExperimentSampleStandard, x_lower:float = -3.0, x_upper:float = 3.0, y_lower:float = -3.0, y_upper:float = 3.0, resolution:int = 128):
    x = jnp.linspace(x_lower, x_upper, resolution)[:, jnp.newaxis]
    y = jnp.linspace(y_lower, y_upper, resolution)[:, jnp.newaxis]

    posterior_predictive_history = []
    posterior_predictive = jnp.zeros((x.shape[0], y.shape[0]))
    means_history = []
    kl_divergences = []

    n = len(experiment.result.samples["parameters"])
    for i in range(n):
        if i % 100 == 0:
            print("processed samples", i)
        sample = experiment.result.samples["parameters"][i]
        std = experiment.result.samples["std"][i]
        means = experiment._model_transformation.apply_from_vector(x, sample)
        means_history.append(means)   
        likelihood_distribution = experiment._model._outputs_likelihood(means, jnp.ones_like(means) * std)
        log_likelihood = likelihood_distribution.log_prob(y.squeeze())
        likelihood = jnp.exp(log_likelihood)
        posterior_predictive += likelihood
        posterior_predictive_history.append(posterior_predictive / (i + 1))
        if i > 0:
            # posterior predictive
            tmp_posterior_predictive_before = posterior_predictive_history[-2]
            tmp_posterior_predictive_after = posterior_predictive_history[-1]
            kl_divergence_value = kl_divergence(
                tmp_posterior_predictive_after,
                tmp_posterior_predictive_before
            )
            kl_divergences.append(kl_divergence_value.mean().item())
    return kl_divergences, posterior_predictive_history

