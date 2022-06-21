# main.py - Created by (c) Gabriel H. Carraretto at 21/04/22

import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import time as t
from itertools import combinations
from functools import reduce
import gurobipy as gp
from gurobipy import GRB
import scipy.sparse as sp

def DBC(graph, S, target):
    """

    :param graph:
    :param S:
    :return:
    """

    n_nodes = graph.number_of_nodes()

    m = gp.Model("DBC")

    # parameter definition
    m.setParam('OutputFlag', 1)
    m.setParam('NonConvex', 2)
    m.setParam('Lazy', 3)
    m.setParam('Presolve', 1)
    m.setParam('TimeLimit', 1800)

    print(f'Problem with {n_nodes} nodes')

    m.setAttr('ModelSense', GRB.MINIMIZE)

    # (6) x_ij in {0, 1}
    x = m.addMVar((n_nodes, n_nodes), ub=1, lb=0, obj=0, vtype=GRB.CONTINUOUS)
    m.update()

    # (8) x_ii = 1 i in S
    x[S, S].obj = 1

    # (7) sum x_ii >= T
    m.addConstr(x.trace() >= target - 1)

    # (2) x_ij <= x_ii forall i in V; j not in N(i)
    for i in range(n_nodes):
        N_bar = list(G.nodes - G.neighbors(i) - {i})
        # not exist k in N_bar(i): (j,k) in E
        for k in N_bar:
            for j in N_bar:
                if k != j:
                    if not graph.has_edge(k, j):
                        m.addConstr(x[i, j] <= x[i, i])

    # (3) sum_{k in N(j), k notin N(i)} x_{ik} >= x_{ii} +x_{jj} -1
    # forall i,j in V: (i,j) notin E
    for i in range(n_nodes):
        for j in range(n_nodes):
            if not graph.has_edge(i, j):
                k = list(set(G.neighbors(j)) - set(G.neighbors(i)))
                m.addConstr(x[i, k].sum() > x[i, i] + x[j, j] - 1)

    # (4) sum_{j notin N(i)} x_ji = 1 forall i in V
    for i in range(n_nodes):
            j = list(G.nodes - G.neighbors(i) - {i})
            if j:
                m.addConstr(x[j, i].sum() == 1)

    # (5) x_ij + x_ik <= x_ii forall i in V; j,k notin N(i); (j,k) in E
    for i in range(n_nodes):
        N_bar = list(G.nodes - G.neighbors(i) - {i})
        for j in N_bar:
            for k in N_bar:
                if G.has_edge(j, k):
                    m.addConstr(x[i, j] + x[i, k] <= x[i, i])

    m.update()
    m.optimize()


def print_graph(graph):
    """
    Given a graph, draw it and show the plot window
    :param graph: a networkx object representing a graph G=(V, E)
    """
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
    TODO: write description
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
        print(start - t.time(), comp_time)
        if start - t.time() > comp_time:
            break
        DBC(graph, S, target)

    return 0


if __name__ == '__main__':

    # test adj. matrix
    A = np.array([
        [0, 0, 0, 1, 0, 0, 0],
        [0, 0, 1, 1, 1, 0, 0],
        [0, 1, 0, 0, 1, 1, 0],
        [1, 0, 0, 0, 1, 0, 0],
        [0, 1, 1, 1, 0, 1, 0],
        [0, 0, 1, 0, 1, 0, 0],
        [1, 0, 0, 0, 0, 0, 0]
    ])

    val = np.array([1.0, 2.0, 3.0, -1.0, -1.0])
    row = np.array([0, 0, 0, 1, 1])
    col = np.array([0, 1, 2, 0, 1])

    # A = sp.csr_matrix((val, (row, col)), shape=(2, 3))

    print(A)
    G = nx.from_numpy_matrix(A)

    solution = iterative_matheuristic_algorithm(
        G,
        target=3,
        policy=non_increasing_ordering,
        comp_time=100
    )
