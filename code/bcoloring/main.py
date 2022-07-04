# main.py - Created by (c) Gabriel H. Carraretto at 21/04/22

import networkx as nx
import numpy as np
import time as t
from itertools import combinations
from DBCmodel import DBCModel


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
    :return: a dictionary of sorted elements containing the {key: degree} of
    each node in the given graph.
    """
    # FIXME: useless sorting
    return dict(sorted(graph.degree, key=lambda x: x[1], reverse=True))


def get_combinations(graph, k):
    """

    :param graph:
    :param k:
    :return:
    """
    return list(combinations(graph.nodes, k))


def select_next_combination(nodes, ranking):
    """

    :param nodes:
    :param ranking:
    :return:
    """
    combination_rankings = {
        node: sum([ranking[elem] for elem in node])
        for node in nodes
    }
    for key in sorted(combination_rankings, key=combination_rankings.get):
        yield key


def iterative_matheuristic_algorithm(graph, target, policy, comp_time):
    """

    :param graph: a networkx object representing a graph G=(V, E)
    :param target: number of colors
    :param policy: policy fn used to rank the b-vertex candidates
    :param comp_time: maximum allowed computational time in seconds
    :return: a b-coloring solution of the graph G with at least the
    number of colors specified by target, or a proof that a coloring
    with target - 1 colors does not exist, or an inconclusive outcome if
    the method runs out of time
    """
    P = get_candidates(graph, target)
    o = policy(P)
    W = get_combinations(P, target)
    next_combination = select_next_combination(W, o)
    start = t.time()
    for S in next_combination:
        # if time limit exceeds, stop the algorithm
        if start - t.time() > comp_time:
            break
        model = DBCModel(graph)
        model.optimize()
    return 0


if __name__ == '__main__':
    # test adj. matrix
    A = np.array([
        [0, 0, 0, 1, 0, 0],
        [0, 0, 1, 1, 1, 0],
        [0, 1, 0, 0, 1, 1],
        [1, 0, 0, 0, 1, 0],
        [0, 1, 1, 1, 0, 1],
        [0, 0, 1, 0, 1, 0],
    ])

    G = nx.from_numpy_matrix(A)

    solution = iterative_matheuristic_algorithm(
        G,
        target=3,
        policy=non_increasing_ordering,
        comp_time=100
    )
