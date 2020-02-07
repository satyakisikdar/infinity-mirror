import pickle

import netgan.utils as utils
from sys import argv
import networkx as nx
import os

def generate(scores, tg_sum, num_graphs):
    graphs = []
    for i in range(num_graphs):
        sparse_mat = utils.graph_from_scores(scores, tg_sum)
        g = nx.from_numpy_array(sparse_mat, create_using=nx.Graph())
        g.name = 'blah'  # filler - renamed later in graph_models.py
        graphs.append(g)
    return graphs


def main():
    if len(argv) < 4:
        print('Needs gname, path to pickle scores and tg_sum, number of graphs')
        exit(1)

    gname, path = argv[1: 3]
    num_graphs = int(argv[3])

    scores, tg_sum = utils.load_pickle(path)
    graphs = generate(scores, tg_sum, num_graphs)

    os.makedirs('./src/netgan/dumps', exist_ok=True)
    pickle.dump(graphs, open(f'./src/netgan/dumps/{gname}_graphs.pkl.gz', 'wb'))
    return

if __name__ == '__main__':
    main()