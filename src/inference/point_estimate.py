import jax
import jax.numpy as jnp
import numpy as np
import optax
from utils import evaluation
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
        parameters = optax.apply_udates(parameters, updates)
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
    loss = (0.5 / jnp.power(std, 2)) * squared_loss + x.shape[0] * jnp.log(std) + 0.5 * regularization
    return loss


class PointEstimate:
    def __init__(self, model_transformation, dataset, optimizer, rng_key):
        self._history = {}
        self._model_transformation = model_transformation
        self._dataset = dataset
        self._parameters = {
            "transformation": self._model_transformation.init(rng_key, self._dataset[0][0]),
            "log_std": 0.0
        }
        self._optimizer = optimizer
        self._optimizer_state = self._optimizer.init(self._parameters)
        self._epochs_counter = 0
    
    def run(self, loss_function, epochs, report_at):
        def loss(params, x, y):
            return loss_function(self._model_transformation, params, x, y)
        loss_compiled = jax.jit(loss)

        inputs_train = self._dataset.data_train[:, self._dataset.conditional_indices]
        outputs_train = self._dataset.data_train[:, self._dataset.dependent_indices]
        inputs_test = self._dataset.data_test[:, self._dataset.conditional_indices]
        outputs_test = self._dataset.data_test[:, self._dataset.dependent_indices]

        for epoch in range(self._epochs_counter, self._epochs_counter + epochs):
            loss_value, gradients = jax.value_and_grad(loss_compiled, 0)(self._parameters, inputs_train, outputs_train)
            updates, self._optimizer_state = self._optimizer.update(gradients, self._optimizer_state)
            self._parameters = optax.apply_updates(self._parameters, updates)
            self._epochs_counter += 1
            if epoch % report_at == 0:
                # save evaluation in history
                loss_value_test = loss_compiled(self._parameters, inputs_test, outputs_test)
                outputs_train_pred = self._model_transformation.apply(self._parameters["transformation"], inputs_train)
                outputs_test_pred = self._model_transformation.apply(self._parameters["transformation"], inputs_test)
                rmse_train = evaluation.rmse(y_true=outputs_train, y_pred=outputs_train_pred)
                rmse_test = evaluation.rmse(y_true=outputs_test, y_pred=outputs_test_pred)
                nll_train = evaluation.nll_gaussian(y_true=outputs_train, y_pred=outputs_train_pred, sigma=jnp.exp(self._parameters["log_std"]))
                nll_test = evaluation.nll_gaussian(y_true=outputs_test, y_pred=outputs_test_pred, sigma=jnp.exp(self._parameters["log_std"]))
                self._history[epoch] = {}
                self._history[epoch]["loss_train"] = loss_value.item()
                self._history[epoch]["loss_test"] = loss_value_test.item()
                self._history[epoch]["rmse_train"] = rmse_train.item()
                self._history[epoch]["rmse_test"] = rmse_test.item()
                self._history[epoch]["nll_train"] = nll_train.item()
                self._history[epoch]["nll_test"] = nll_test.item()

                # print output
                print(f"epoch {epoch} loss_train: {loss_value} loss_test: {loss_value_test}")
    def save(self, fn_name):
        result_dictionary = {
            "history": self._history,
        }
    
    
    @property
    def parameters(self):
        return self._parameters
    
    @property
    def parameters_serializable(self):
        parameters_vector = jax.tree_util.tree_reduce(lambda a, b: jnp.concatenate([a.flatten(), b.flatten()]), self._parameters["transformation"])[jnp.newaxis]
        d = {
            "transformation": np.array(parameters_vector).tolist(),
            "log_std": np.array(self._parameters["log_std"]).tolist()
        }
        return d

    @property
    def transformation_parameters_vector(self):
        return jax.tree_util.tree_reduce(lambda a, b: jnp.concatenate([a.flatten(), b.flatten()]), self._parameters["transformation"])[jnp.newaxis]

    @property
    def history(self):
        return self._history

    @property
    def parameters(self):
        return self._parameters
