import pickle
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from random import random
from networkx.drawing.nx_agraph import graphviz_layout

def f(x):
    if 0 <= x < 10/3:
        return 2
    elif 10/3 <= x < 2*10/3:
        return 3
    else:
        return 4

def main():
    G = nx.Graph()
    ctr = 1
    G.add_node(1)
    height = [1]
    while len(G) <= 1000:
        new_nodes = []
        for u in height:
            r = f(10*random()) # r ∈ {2, 3, 4)}
            nodes = [ctr + i + 1 for i in range(r)]
            edges = [(u, v) for v in nodes]

            G.add_nodes_from(nodes)
            G.add_edges_from(edges)

            new_nodes += nodes
            ctr += r
        height = new_nodes

    if len(G.nodes) == len(G.edges) + 1 and nx.is_connected(G):
        nx.draw(G, node_size=10, alpha=0.5)
        plt.show()
        print(len(G))
        nx.write_edgelist(G, '../input/tree.g', data=False)
    else:
        print('wew lad that\'s not a tree')

main()
