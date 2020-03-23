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

base_path = '/home/danielgonzalez/repos/infinity-mirror/output/pickles'
graph_type = 'clique-ring-25-4'
model_name = 'CNRG'

path = os.path.join(base_path, graph_type, model_name)

for filename in os.listdir(path):
    print(f'starting\t{filename}')

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

    print(f'done with\t{filename}')

exit()

for subdir, dirs, files in os.walk(data_path):
    tail = subdir[30:]
    for file in files:
        if 'fast' in file:
            infile = os.path.join(data_path, tail, file)
            outfile = os.path.join(base_path, tail, file)
            print(f'start\t{infile}')
            try:
                old_root = load_pickle(infile)
                new_root = TreeNode(name=old_root.name,\
                                    stats=old_root.stats,\
                                    stats_seq={},\
                                    graph=old_root.graph)
                node = new_root
                while len(old_root.children) > 0:
                    old_root = old_root.children[0]
                    node = TreeNode(name=old_root.name,\
                                    stats=old_root.stats,\
                                    graph=old_root.graph,\
                                    parent=node)
                with open(outfile, 'wb') as f:
                    pickle.dump(new_root, f)
                print(f'done\t{outfile}')
            except ModuleNotFoundError:
                print(f'ERROR: {infile}')
