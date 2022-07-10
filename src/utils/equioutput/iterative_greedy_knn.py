import numpy as np
import jax.numpy as jnp


class IterativeGreedyKNN:
    def __init__(self):
        pass
    
    def run(self, samples_subspace, labels, similarity_matrix, indices=None, dense=True):
        n, hidden, dim = samples_subspace.shape
        assert labels.shape == (n, hidden)
        if indices is None:
            indices = np.arange(n)
        
        labels_result = labels.copy()
        x = samples_subspace.reshape((-1, dim))
        y = labels.reshape((-1, ))

        for i in indices:
            hneurons_probs = np.zeros((hidden + 1, hidden)) #  why +1?
            hneurons_probs[0] = np.arange(hidden)
            for h in range(hidden):
                element = i * hidden + h
                if dense:
                    similarities = np.array(similarity_matrix[element]).squeeze()
                else:
                    similarities = np.array(similarity_matrix[element].todense()).reshape((-1, ))
                selection = (y == np.arange(hidden)[:, np.newaxis])
                weights = np.einsum("j, ij -> i", similarities, selection)
                hneurons_probs[1 + h] = weights / weights.sum()
            
            labels_current = np.zeros((hidden), dtype=np.int32)
            #print(hneurons_probs)
            for h in range(hidden):
                max_nodes_probs = jnp.max(hneurons_probs[1:], axis=1)
                max_prob_node = jnp.argmax(max_nodes_probs)
                max_prob_node_max_prob_label = jnp.argmax(hneurons_probs[1 + max_prob_node])
                #print(max_nodes_probs, max_prob_node, max_prob_node_max_prob_label)
                labels_current[max_prob_node] = max_prob_node_max_prob_label
                hneurons_probs[1 + max_prob_node, :] = -1.0
                hneurons_probs[:, max_prob_node_max_prob_label] = -1.0
            labels_result[i] = labels_current
            #print("in", labels_current)
        
        return labels_result[indices], indices