import jax
import jax.numpy as jnp
import optax
from data.datasets import ConditionalDataset


def point_estimate(transformation, loss_function, parameters, dataset, epochs, optimizer, report_at):
    def loss(params, x, y):
        return loss_function(transformation, params, x, y)
    loss_compiled = jax.jit(loss)

    optimizer_state = optimizer.init(parameters)
    inputs = dataset.data[:, dataset.conditional_indices]
    outputs = dataset.data[:, dataset.dependent_indices]
    for epoch in range(epochs):
        loss_value, gradients = jax.value_and_grad(loss_compiled, 0)(parameters, inputs, outputs)
        updates, optimizer_state = optimizer.update(gradients, optimizer_state)
        parameters = optax.apply_updates(parameters, updates)
        if epoch % report_at == 0:
            print(f"epoch {epoch} loss: {loss_value}")
    return parameters, optimizer_state


def ridge_loss(transformation, params, x, y):
    std = jnp.exp(params["log_std"])
    y_pred = transformation.apply(params["transformation"], x)
    squared_loss = jnp.power(y - y_pred, 2).sum()
    regularization = jax.tree_util.tree_reduce(
        lambda a, b: a + b,
        jax.tree_util.tree_map(lambda a: jnp.power(a, 2).sum(), params["transformation"])
    )
    loss = (0.5 / jnp.power(std, 2)) * squared_loss + x.shape[0] * jnp.log(std) + regularization
    return loss
