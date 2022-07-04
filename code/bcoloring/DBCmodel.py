# DBCmodel.py - Created by (c) Gabriel H. Carraretto at 29/06/22
import gurobipy as gp
from gurobipy import GRB
from itertools import combinations
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.colors
import numpy as np


class DBCModel:
    def __init__(self, graph):
        """
        Initialize the model variables
        :param graph: nx-Graph obj
        """
        self.graph = graph
        self.model = gp.Model("DBC Model")
        self.x = None

    def create_model(self):
        """
        Creates the DBC model by applying the constraints imposed by
        https://arxiv.org/pdf/2102.09696.pdf and
        https://dl.acm.org/doi/pdf/10.1145/3524338.3524379
        :return: None
        """

        n_nodes = self.graph.number_of_nodes()

        # parameter definition
        self.model.setParam('OutputFlag', 0)
        self.model.setParam('NonConvex', 2)
        self.model.setParam('Lazy', 3)
        self.model.setParam('Presolve', 1)
        self.model.setParam('TimeLimit', 1800)

        self.model.setAttr('ModelSense', GRB.MINIMIZE)

        # x_uv equal to one if vertex u represents the color of vertex v
        # and zero otherwise, defined for every ordered pair (u,v), with
        # u ∈ V and v ∈ N̅̅[u]

        x = self.model.addMVar(
            (n_nodes, n_nodes),
            ub=1,
            lb=0,
            obj=0,
            vtype=GRB.BINARY,
            name='x'
        )

        self.model.update()

        for u in range(n_nodes):

            neighbors = list(self.graph.neighbors(u))  # N(u)
            n_bar: list = list(self.graph.nodes - neighbors - {u})  # N̅(u)
            n_bar_closed = n_bar + [u]  # N̅[u]

            # (6) x_uv in {0, 1} ∀ i ∈ V, v ∈  N̅[u]
            # ensure that the neighbors cannot be represented by vertex u
            self.model.addConstr(x[u, neighbors] == 0)
            self.model.addConstr(x[neighbors, u] == 0)
            if n_bar_closed:
                # (2) sum_{v ∈ N̅[u]} x_vu = 1 ∀ u ∈ V
                # ensure that every vertex must have a color
                self.model.addConstr(x[n_bar_closed, u].sum() == 1)

                connected_set = set()

                for v, w in combinations(iterable=n_bar, r=2):
                    if self.graph.has_edge(v, w):
                        # (3) x_uv + x_uw <= x_uu ∀ i ∈ V;
                        # v, w ∈ N̅(u); (v, w) ∈ E
                        # force the coloring to be proper
                        self.model.addConstr(x[u, v] + x[u, w] <= x[u, u])

                        # update the connected set in the
                        # anti-neighborhood of vertex u
                        connected_set.update({v, w})

                # set of vertices in the anti-neighborhood of u and not
                # connected on it
                n_bar_star = list(set(n_bar) - connected_set)  # N̅*(u)

                # (4) x_uv <= x_uu ∀ i ∈ V; j ∈ N̅*(u)
                # guarantee that a vertex can only give a color if it is
                # a representative
                for v in n_bar_star:
                    self.model.addConstr(x[u, v] <= x[u, u])

            # (5) x_uv <= x_uu ∀ i ∈ V; j ∈ N̅*(u)
            # if both u and v are b-vertices,then there must be a
            # neighbor of v which is represented by u
            for v in n_bar:
                # N(v) ∩ N̅(u)
                w_set = list(set(self.graph.neighbors(v)) - set(n_bar))
                self.model.addConstr(
                    x[u, w_set].sum() >= x[u, u] + x[v, v] - 1
                )

        self.model.update()
        self.x = x

    def optimize(self, S, target, verbose=2):
        """
        Given a set of possible b-vertices and an initial target T
        :param S: set of possible b-vertices
        :param target: Integer representing the initial target
        :param verbose: flag used for output suppression
        :return:
        """
        if not 0 <= verbose <= 2:
            raise ValueError("verbose must be between 0 and 2")

        # remove previous constraints
        for i in range(len(S)):
            c = self.model.getConstrByName('constr8.' + str(i))
            if c:
                self.model.remove(c)

        self.model.update()

        # (8) x_uu = 1 i ∈ S
        # Set to one the variables corresponding to the b-vertices
        # imposed by the heuristic
        for i, u in enumerate(S):
            self.model.addConstr(self.x[u, u] == 1, name='constr8.' + str(i))

        self.model.update()

        n_nodes = self.graph.number_of_nodes()

        c = self.model.getConstrByName('constr7')
        if c:
            self.model.remove(c)

        self.model.update()

        # (7) sum x_uu >= T
        # Only feasible solutions with at least T (target) b-vertices
        self.model.addConstr(
            self.x[range(n_nodes), range(n_nodes)].sum() >= target,
            name='constr7'
        )

        self.model.optimize()

        if self.model.Status == GRB.OPTIMAL:
            if verbose == 2:
                self.print_graph(solution_matrix=self.x.X)
            res = (self.model.Runtime, self.model.getJSONSolution())
        else:
            res = (self.model.Runtime, {})
        self.model.reset()
        return res

    def print_graph(self, solution_matrix=None):
        """
        Given a graph, draw it and show the plot window
        :param solution_matrix: optional solution matrix
        """
        if solution_matrix is not None:
            colors = self.from_sol_to_colors(solution_matrix)
            nx.draw(self.graph, node_color=colors, with_labels=True)
        else:
            nx.draw(self.graph, with_labels=True)
        plt.show()

    def from_sol_to_colors(self, sol_mat):
        """
        Given a solution matrix X it returns the colors for each vertex
        :param sol_mat: NxN np Matrix
        :return: array of color values
        """
        number_of_colors = sol_mat.trace().sum()
        n_nodes = sol_mat.shape[0]
        colors = {i: [] for i in range(n_nodes)}
        cmap = matplotlib.cm.get_cmap(name="gist_rainbow")
        j = 0
        for i in range(n_nodes):
            if sol_mat[i, i] == 1:
                same_color = np.argwhere(sol_mat[i, :] == 1)
                for node in same_color:
                    colors[int(node)] = cmap(float(j / int(number_of_colors)))
                j += 1
        return [colors[j] for j in range(n_nodes)]
