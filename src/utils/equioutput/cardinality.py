import math


def tanh_cardinality(hidden):
    return 2**hidden


def permutation_cardinality(hidden):
    return math.factorial(hidden)


def tanh_layer_cardinality(hidden):
    return permutation_cardinality(hidden) * tanh_cardinality(hidden)


def tanh_mlp_cardinalities(hidden_layers_sizes):
    permutation = 1
    tanh = 1
    total = 1
    for layer_size in hidden_layers_sizes:
        permutation *= permutation_cardinality(layer_size)
        tanh *= tanh_cardinality(layer_size)
        total *= tanh_layer_cardinality(layer_size)
    return permutation, tanh, total

