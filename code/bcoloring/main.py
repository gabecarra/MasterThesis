# main.py - Created by (c) Gabriel H. Carraretto at 21/04/22

import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import time as t
import gurobipy as gp


def print_graph(graph):
    nx.draw(graph, with_labels=True)
    plt.show()


def get_candidates(graph, target):
    """
    Creates a subgraph containing all the vertices that has at least
    target - 1 number of neighbors, which then can be considered as
    candidates to be b-vertices.
    :param graph: a networkx object representing a graph G=(V, E)
    :param target: number of colors
    :return: subgraph of possible b-vertices
    """
    candidates = [node for (node, val) in graph.degree if val >= target - 1]
    return graph.subgraph(candidates)


def non_increasing_ordering(graph):
    """
    Sorts the given graph nodes based on the number of neighbors.
    :param graph: a networkx object representing a graph G=(V, E).
    :return: an array of sorted tuples containing the (key, degree) of
    each node in the given graph.
    """
    return sorted(graph.degree, key=lambda x: x[1], reverse=True)


# TODO: write fn
def get_combinations(graph):
    return 0


def iterative_matheuristic_algorithm(graph, target, policy, comp_time):
    """
    TODO: write description
    :param graph: a networkx object representing a graph G=(V, E)
    :param target: number of colors
    :param policy: policy used to rank the b-vertex candidates
    :param comp_time: maximum allowed computational time in seconds
    :return: a b-coloring solution of the graph G with at least the
    number of colors specified by target, or a proof that a coloring
    with target - 1 colors does not exist, or an inconclusive outcome if
    the method runs out of time
    """
    P = get_candidates(graph, target)
    o = policy(P)
    W = get_combinations(P)
    start = t.time()
    while W and start - t.time() < comp_time:
        # TODO: create model
        pass
    return 0


if __name__ == '__main__':

    # test adj. matrix
    A = np.array([
        [0, 0, 0, 1, 0, 0],
        [0, 0, 1, 1, 1, 0],
        [0, 1, 0, 0, 1, 1],
        [1, 0, 0, 0, 1, 0],
        [0, 1, 1, 1, 0, 1],
        [0, 0, 1, 0, 1, 0]
    ])

    G = nx.from_numpy_matrix(A)

    solution = iterative_matheuristic_algorithm(
        G,
        target=2,
        policy=non_increasing_ordering,
        comp_time=100
    )
