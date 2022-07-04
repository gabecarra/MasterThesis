# graph_parse.py - Created by (c) Gabriel H. Carraretto at 30/06/22
import os
import networkx as nx
import matplotlib.pyplot as plt
from tqdm import tqdm


def get_graph_instance(root):
    """
    Given a root path, iterates through all the folders in that path
    while reading each graph instance and yields each graph as an edge
    list [[e_u, e_k], ... [e_p, e_w]]
    :param root: string representing a path
    :return: tuple (filename, edge list)
    """
    # for each folder in root
    for path, _, files in os.walk(root):
        # for each file in a given folder
        for file in files:
            if file.endswith(('.col', 'clq')):
                print(f'Reading instance {file}')
                file_path = path + '/' + file
                num_lines = sum(1 for line in open(file_path, 'r'))
                edgelist = [
                    list(map(lambda val: val - 1, map(int, x.split()[1:3])))
                    for x in tqdm(
                        open(file_path, 'r').readlines(), total=num_lines
                    )
                    if x.startswith('e')
                ]
                yield file.split('.c')[0], edgelist


if __name__ == '__main__':
    root = os.path.join(os.path.dirname(__file__),
                        'instances/')
    for path, _, files in os.walk(root):
        # for each file in a given folder
        for file in files:
            if file.endswith(('.col', 'clq')):
                file_path = path + '/' + file
                edgelist = [
                    int(elem)
                    for x in open(file_path, 'r', encoding="ISO-8859-1").readlines()
                    for elem in x.split()[1:3]
                    if x.startswith('e')
                ]
                if not edgelist:
                    print(file)
    pass
