import time
import jax
import jax.numpy as jnp
import optax


def point_estimate(loss_function, parameters, inputs, outputs, epochs, optimizer, optimizer_state, report_at):
    for epoch in range(epochs):
        loss_value, gradients = jax.value_and_grad(loss_function, 0)(parameters, inputs, outputs)
        updates, optimizer_state = optimizer.update(gradients, optimizer_state)
        parameters = optax.apply_updates(parameters, updates)
        if epoch % report_at == 0:
            print(f"epoch {epoch} loss: {loss_value}")
    return parameters, optimizer_state

