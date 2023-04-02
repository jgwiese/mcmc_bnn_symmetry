import jax
import jax.numpy as jnp
from numpyro import distributions


def computed_lppd_mcmc(inputs, outputs, parameters_network, parameters_data_std, regression_model):
    """variant of lppd for mcmc samples"""
    log_prob_means = []
    for xi, yi in zip(inputs, outputs):
        yi_preds = jax.vmap(regression_model._transformation.apply_from_vector, in_axes=(None, 0))(xi, parameters_network)
        prob_mean = jnp.exp(regression_model._outputs_likelihood(
            yi_preds,
            parameters_data_std).log_prob(yi)).mean(axis=0)
        if prob_mean > 0.0:
            log_prob_means.append(jnp.log(prob_mean))
    log_prob_means = jnp.array(log_prob_means)
    return log_prob_means

def computed_lppd_la(inputs, outputs, parameters_network, parameters_data_std, regression_model):
    """variant of lppd for la samples"""
    log_prob_means = []
    for xi, yi in zip(inputs, outputs):
        yi_preds = jax.vmap(regression_model._transformation.apply_from_vector, in_axes=(None, 0))(xi, parameters_network)
        yi_std = parameters_data_std
        prob_mean = jnp.exp(regression_model._outputs_likelihood(
            yi_preds,
            yi_std).log_prob(yi)).mean(axis=0)
        if prob_mean > 0.0:
            log_prob_means.append(jnp.log(prob_mean))
    log_prob_means = jnp.array(log_prob_means)
    return log_prob_means

def computed_lppd_de(inputs, outputs, parameters_network, parameters_data_std, regression_model):
    """variant of lppd for de samples"""
    log_prob_means = []
    for xi, yi in zip(inputs, outputs):
        yi_preds = jax.vmap(regression_model._transformation.apply_from_vector, in_axes=(None, 0))(xi, parameters_network)
        # means
        mean = yi_preds.mean(0)
        variance = (jnp.power(parameters_data_std, 2) + jnp.power(yi_preds, 2)).mean(0) - jnp.power(mean, 2)
        std = jnp.power(variance, 0.5)
        predictive_prob = jnp.exp(distributions.Normal(mean, std).log_prob(yi))
        log_prob_means.append(jnp.log(predictive_prob))
    log_prob_means = jnp.array(log_prob_means)
    return log_prob_means

