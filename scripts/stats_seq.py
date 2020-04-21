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

base_path = '/home/danielgonzalez/repos/infinity-mirror/output/pickles/eucore'
model = 'Linear_AE'

path = os.path.join(base_path, model)

for subdir, dirs, files in os.walk(path):
    for filename in files:
        string = subdir.split('/')[-2:]
        file = os.path.join(subdir, filename)
        print(f'starting\t{string[-2]}\t{string[-1]}\t{filename} ...')
        root = load_pickle(file)
        node = root
        if node.stats_seq is None or node.stats_seq == {}:
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
                child.stats_seq['degree_cvm'] = comparator.cvm_degree()
                child.stats_seq['pagerank_cvm'] = comparator.cvm_pagerank()
                node = child
        with open(file, 'wb') as f:
            pickle.dump(root, f)
        print(f'\tdone')
