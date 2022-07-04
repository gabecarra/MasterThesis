# policy.py - Created by (c) Gabriel H. Carraretto at 29/06/22


def non_increasing_ordering(vertices, graph):
    """
    Sorts the given graph nodes based on the number of neighbors.
    :param graph: a networkx object representing a graph G=(V, E).
    :return: a dictionary of sorted elements containing the {key: degree} of
    each node in the given graph.
    """
    b_vertices = {key: graph.degree[key] for key in vertices}
    return sorted(b_vertices, key=b_vertices.get)
