import numpy as np
import sys
import pickle as pkl
import networkx as nx
import scipy.sparse as sp


def parse_index_file(filename):
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index

'''
    modified data reader to take in adjacency matrices
    backwards compatible with the original load_data(dataset)
'''
def load_data(dataset, feature_flag=1):
    if dataset not in ['cora', 'citeseer', 'pubmed']:
        # try to load <dataset>.g and take only the largest connected component
        G = nx.read_adjlist('data/{}.g'.format(dataset), nodetype=int)
        graph = max(nx.connected_component_subgraphs(G), key=len)
        adj = nx.adjacency_matrix(graph, nodelist=sorted(graph.nodes()))
        return adj, sp.identity(adj.shape[0])
    else:
        # load the data: x, tx, allx, graph
        names = ['x', 'tx', 'allx', 'graph']
        objects = []
        for i in names:
            with open('data/ind.{}.{}'.format(dataset, i), 'rb') as f:
                if sys.version_info > (3, 0):
                    objects.append(pkl.load(f, encoding='latin1'))
                else:
                    objects.append(pkl.load(f))
        x, tx, allx, graph = tuple(objects)
        test_idx_reorder = parse_index_file('data/ind.{}.test.index'.format(dataset))
        test_idx_range = np.sort(test_idx_reorder)
        if dataset == 'citeseer':
            # Find isolated nodes, add them as zero-vecs into the right position
            test_idx_range_full = range(min(test_idx_reorder), max(test_idx_reorder)+1)
            tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
            tx_extended[test_idx_range-min(test_idx_range), :] = tx
            tx = tx_extended
        if feature_flag:
            features = sp.vstack((allx, tx)).tolil()
            features[test_idx_reorder, :] = features[test_idx_range, :]
        else:
            features = sp.identity(features.shape[0])  # featureless
        adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))
    return adj, features
