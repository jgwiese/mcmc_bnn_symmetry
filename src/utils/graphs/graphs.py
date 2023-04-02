import jax.numpy as jnp
import scipy.sparse as sp
import numpy as np
from tqdm import tqdm


def p_norm(v, p):
    return jnp.power(jnp.sum(jnp.power(jnp.abs(v), p), axis=-1), 1.0/p)


def similarity(v, sigma=1.0):
    return jnp.exp(- (1.0 / 2.0 * jnp.power(sigma, 2)) * v)


def distance_knn_graph(nodes, k=1):
    k = int(k)
    adjacency_matrix = sp.lil_matrix((len(nodes), len(nodes)), dtype=jnp.float32)
    for i, node in enumerate(tqdm(nodes)):
        distances = p_norm(nodes - node, p=2.0)
        nearest_neighbors_indices = jnp.argsort(distances)
        k_selection = nearest_neighbors_indices[1:1 + k]
        adjacency_matrix[i, k_selection] = distances[k_selection]
        adjacency_matrix[k_selection, i] = distances[k_selection]
    return adjacency_matrix.tocsr()


def distance_knn_graph_dense(nodes, k=1):
    k = int(k)
    adjacency_matrix = np.zeros((len(nodes), len(nodes)), dtype=np.float32)
    for i, node in enumerate(tqdm(nodes)):
        distances = p_norm(nodes - node, p=2.0)
        nearest_neighbors_indices = jnp.argsort(distances)
        k_selection = nearest_neighbors_indices[1:1 + k]
        adjacency_matrix[i, k_selection] = distances[k_selection]
        adjacency_matrix[k_selection, i] = distances[k_selection]
    return adjacency_matrix


def knn_graph(nodes, k=None, sigma=1.0):
    """ Nodes have features: position vector coordinates. """
    n, dim = nodes.shape
    if k is None:
        k = n * dim
    k = int(k)
    adjacency_matrix = sp.lil_matrix((len(nodes), len(nodes)), dtype=jnp.float32)
    for i, node in enumerate(tqdm(nodes)):
        distances = p_norm(nodes - node, p=2.0)
        nearest_neighbors_indices = jnp.argsort(distances)
        k_selection = nearest_neighbors_indices[1:1 + k]
        s = similarity(distances[k_selection], sigma=sigma)
        adjacency_matrix[i, k_selection] = s
        adjacency_matrix[k_selection, i] = s
    return adjacency_matrix.tocsr()


def knn_graph_dense(nodes, k=None, sigma=1.0):
    """ Nodes have features: position vector coordinates. """
    n, dim = nodes.shape
    if k is None:
        k = n * dim
    k = int(k)
    adjacency_matrix = np.zeros((len(nodes), len(nodes)), dtype=np.float32)
    for i, node in enumerate(tqdm(nodes)):
        distances = p_norm(nodes - node, p=2.0)
        nearest_neighbors_indices = jnp.argsort(distances)
        k_selection = nearest_neighbors_indices[1:1 + k]
        s = similarity(distances[k_selection], sigma=sigma)
        adjacency_matrix[i, k_selection] = s
        adjacency_matrix[k_selection, i] = s
    return adjacency_matrix


def special_graph(nodes, hidden, sigma=1.0):
    print(nodes.shape)
    n = int(len(nodes) / hidden)
    print(n)
    adjacency_matrix = sp.lil_matrix((len(nodes), len(nodes)), dtype=jnp.float32)
    for i in tqdm(range(n)):
        index = int(i * hidden)
        for j in range(0, hidden):
            for k in range(0, hidden):
                index_1 = index + j
                index_2 = index + k
                if index_1 != index_2:
                    distance = p_norm(nodes[index_1] - nodes[index_2], p=2.0)
                    similarity = jnp.exp(- (1.0 / (2.0 * sigma)) * distance)
                    adjacency_matrix[index_1, index_2] = similarity
    return adjacency_matrix.tocsr()


def degree_matrix(a, signed=False):
    if signed:
        elements = jnp.asarray(jnp.abs(a).sum(0)).squeeze()
    else:
        elements = jnp.asarray(a.sum(0)).squeeze()
    degree_matrix = sp.csr_matrix(
        (elements,
        (jnp.arange(a.shape[0], dtype=jnp.int32), jnp.arange(a.shape[0], dtype=jnp.int32))),
        shape=a.shape,
        dtype=jnp.float32
    )
    return degree_matrix


def laplacian(a, d, normalized=True):
    if normalized:
        l = sp.eye(a.shape[0]) - (d.power(- 0.5) @ a @ d.power(- 0.5))
    else:
        l = d - a
    return l


def spectrum(l, k, normalized=True):
    eigenvalues, eigenvectors = sp.linalg.eigsh(l, k=k, which="SM")
    lengths = p_norm(eigenvectors, p=2)[:, jnp.newaxis]
    if normalized:
        eigenvalues = eigenvalues / eigenvalues.sum()
        eigenvectors = eigenvectors / lengths
    return eigenvalues, eigenvectors


def get_knn_sc_eigenvalues(samples, n=256):
    a = knn_graph(nodes=samples, k=4)
    d = degree_matrix(a)
    l = laplacian(a=a, d=d, normalized=True)
    eigenvalues, eigenvectors = spectrum(l=l, k=n, normalized=False)
    return eigenvalues, eigenvectors

