# IM_algorithm.py - Created by (c) Gabriel H. Carraretto at 29/06/22

import time as t
from itertools import combinations
from DBCmodel import DBCModel
from math import comb

class IterativeMatheuristicAlgorithm:
    def __init__(self, graph, target, policy, comp_time=180):
        self.graph = graph
        self.target = target
        self.policy = policy
        self.comp_time = comp_time

    def get_candidates(self):
        """
        Creates a subgraph containing all the vertices that has at least
        target - 1 number of neighbors, which then can be considered as
        candidates to be b-vertices.
        :return: list of possible b-vertices
        """
        candidates = [
            node for (node, val) in self.graph.degree if val >= self.target - 1
        ]
        return candidates

    def select_next_combination(self, ranking):
        """
        Given the rankings of each node, sorts it, and yields the
        node combinations s.t. argmin(sum_{i in W} o(i))
        :param ranking: dictionary of sorted rankings {node:rank, ...}
        :return: a combination of nodes of size T s.t. their ranking sum
        is minimized respect to all the possible combinations
        """
        for combination in combinations(ranking, self.target):
            yield combination

    def run(self, verbose=2):
        """
        Performs the Iterative Matheuristic Algorithm (IMA)
        :return: a b-coloring solution of the graph G with at least the
        number of colors specified by target, or a proof that a coloring
        with target - 1 colors does not exist, or an inconclusive outcome if
        the method runs out of time
        """

        model = DBCModel(self.graph)
        print('Creating model')
        model.create_model()

        run_time = 0
        best_solution = {}
        best_run_time = 0

        while run_time < self.comp_time:
            print(f'Optimizing problem with '
                  f'n_nodes={self.graph.number_of_nodes()},'
                  f' T={self.target} and '
                  f'tot. combinations={comb(self.graph.number_of_nodes(), self.target)}...',
                  end='')
            P = self.get_candidates()
            o = self.policy(P, self.graph)
            get_combination = self.select_next_combination(o)
            while True:
                try:
                    S = next(get_combination)
                    sol_time, solution = model.optimize(S, self.target,
                                                        verbose)
                    run_time += sol_time

                    # if there is a solution, prints an ack, increments
                    # the target, resets the runtime and breaks the loop
                    if solution:
                        print("✅ ")
                        self.target += 1
                        best_solution = solution
                        best_run_time = run_time
                        run_time = 0
                        break

                except StopIteration:
                    # breaks in case there are no more combinations
                    print("❌ ")
                    print(f"No more combinations possible. "
                          f"Solution is infeasible for T={self.target}")
                    return t.strftime('%H:%M:%S', t.gmtime(best_run_time)), \
                        best_solution

        # if time limit exceeds, or no solutions are possible for the
        # given target, stops the algorithm
        print("❌ ")
        print(f"Inconclusive computation due to time limit constraints.")
        return t.strftime('%H:%M:%S', t.gmtime(best_run_time)), best_solution
