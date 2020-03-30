import os
import sys; sys.path.append('./..')
import pickle
import networkx as nx # requires 2.3.0
import pandas as pd; pd.options.display.float_format = '{:,.2f}'.format
import statsmodels.stats.api as sm
import warnings; warnings.filterwarnings("ignore", category=UserWarning)
from glob import glob
from statistics import median_low

from src.Tree import TreeNode
from src.utils import load_pickle
from src.graph_stats import GraphStats
from src.graph_comparison import GraphPairCompare

graphs = ['3-comm', 'BA-1000-3', 'BA-100-3', 'clique-ring-100-4', 'clique-ring-25-4', 'clique-ring-50-4', 'dolphins', 'eucore', 'flights', 'football', 'grid-100-100', 'grid-10-10', 'karate', 'ladder-10', 'ladder-100', 'ladder-20', 'ladder-4', 'ladder-50', 'ladder-500', 'ring-10', 'ring-100', 'ring-1000', 'ring-20', 'ring-500']
models = ['BTER', 'Chung-Lu', 'CNRG', 'Erdos-Renyi', 'GraphAE', 'GraphVAE', 'HRG', 'NetGAN', 'SBM']

base_path = '/home/danielgonzalez/repos/infinity-mirror/output/pickles'
sel_graphs = ['clique-ring-25-4']
sel_models = models[::]

for graph in sel_graphs:
    for model in sel_models:
        path = os.path.join(base_path, graph, model)
        print(f'starting {graph}/{model}')
        for filename in os.listdir(path):
            print(f'\tstarting\t{filename} ...', end='')
            root = load_pickle(os.path.join(path, filename))
            node = root
            node.stats_seq = {}
            while len(node.children) > 0:
                child = node.children[0]
                comparator = GraphPairCompare(GraphStats(graph=node.graph, run_id=0),\
                                              GraphStats(graph=child.graph, run_id=0))
                child.stats_seq = {}
                child.stats_seq['lambda_dist'] = comparator.lambda_dist()
                child.stats_seq['node_diff'] = comparator.node_diff()
                child.stats_seq['edge_diff'] = comparator.edge_diff()
                child.stats_seq['pgd_pearson'] = comparator.pgd_pearson()
                child.stats_seq['pgd_spearman'] = comparator.pgd_spearman()
                child.stats_seq['deltacon0'] = comparator.deltacon0()
                child.stats_seq['cvm_degree'] = comparator.cvm_degree()
                child.stats_seq['cvm_pagerank'] = comparator.cvm_pagerank()
                node = child
            with open(os.path.join(path, filename), 'wb') as f:
                pickle.dump(root, f)

            print(f'\tdone')
        print(f'done with {graph}/{model}')
