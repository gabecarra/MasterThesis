# generate_dataset.py - Created by (c) Gabriel H. Carraretto at 29/06/22

from im_algorithm import IterativeMatheuristicAlgorithm
from policy import non_increasing_ordering
from networkx.readwrite import json_graph
import json
import matplotlib.pyplot as plt
import numpy as np
from graph_parse import get_graph_instance
import networkx as nx
import os

if __name__ == '__main__':
    root = os.path.join(os.path.dirname(__file__), 'instances/asd')
    for name, edge_list in get_graph_instance(root):

        G = nx.from_edgelist(edge_list)
        G.name = name

        # test adj. matrix
        # A = np.array([
        #     [0, 0, 0, 1, 0, 0],
        #     [0, 0, 1, 1, 1, 0],
        #     [0, 1, 0, 0, 1, 1],
        #     [1, 0, 0, 0, 1, 0],
        #     [0, 1, 1, 1, 0, 1],
        #     [0, 0, 1, 0, 1, 0],
        # ])
        #
        # G = nx.from_numpy_matrix(A)

        print(f'Instance name: {name}, size: {G.number_of_nodes()}')
        # nx.draw(G)
        # plt.show()

        im_algorithm = IterativeMatheuristicAlgorithm(
            G,
            target=2,
            policy=non_increasing_ordering,
            comp_time=180
        )

        exec_time, solution = im_algorithm.run(verbose=1)

        with open(f'results/{name}.json', 'w') as outfile:
            res = dict(
                execTime=exec_time,
                graph=json_graph.adjacency_data(G),
                solution=json.loads(solution) if solution else {}
            )
            json.dump(res, outfile)
