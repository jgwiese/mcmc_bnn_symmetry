import jax
import jax.numpy as jnp


def loss_svm_unsupervised(params, inputs):
    hyperplane = jnp.sqrt(params["normal"].T @ params["normal"])
    constraints = jnp.sum(jnp.maximum(0.0, 1.0 - jnp.abs(inputs @ params["normal"])))
    c = 1.0
    return hyperplane + c * constraints


class UnsupervisedSVMBinary:
    def __init__(self, samples_subspace, rng_key=jax.random.PRNGKey(0)):
        self._samples_subspace = samples_subspace
        self._rng_key, rng_key_ = jax.random.split(rng_key)
        self._hyperplane_params = {
            "normal": jax.random.normal(rng_key_, (self._samples_subspace.shape[-1], ))
        }
        self._hyperplane_params["normal"] /= jnp.linalg.norm(self._hyperplane_params["normal"])
        self._current_epoch = 0
        self._history = {}

    @property
    def normal(self):
        return self._hyperplane_params["normal"]

    def _lr_decay(self, lr):
        return 0.8 * lr

    def optimize(self, epochs, batch_size, lr, report_at):
        loss_compiled = jax.jit(loss_svm_unsupervised)
        self._history[0] = [
            self._hyperplane_params["normal"],
            loss_compiled(self._hyperplane_params, self._samples_subspace)
        ]

        for epoch in range(self._current_epoch, self._current_epoch + epochs):
            self._rng_key, key_batch = jax.random.split(self._rng_key)
            indices = jax.random.permutation(key_batch, jnp.arange(len(self._samples_subspace)))
            batches = int(len(indices) / batch_size)
            
            batch_values = []
            for b in range(batches):
                batch_indices = indices[b * batch_size:(b + 1) * batch_size]
                batch = self._samples_subspace[batch_indices]
                value, grad = jax.value_and_grad(loss_compiled)(self._hyperplane_params, batch)
                batch_values.append(value)
                self._hyperplane_params["normal"] = self._hyperplane_params["normal"] - lr * grad["normal"]
            value = sum(batch_values) / batches
            self._history[epoch] = [self._hyperplane_params["normal"], value]
            lr = self._lr_decay(lr)

            if epoch % report_at == 0:
                print(f'epoch: {epoch},  loss: {value}, normal l2 norm: {jnp.linalg.norm(self._hyperplane_params["normal"])}')
        
        self._current_epoch += epochs
