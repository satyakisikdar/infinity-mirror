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

graphs = ['eucore', 'clique-ring-500-4', 'flights', 'treer1000']

base_path = '/data/dgonza26'
dataset = 'eucore'
#model = 'BTER'
models = ['BUGGE', 'Chung-Lu', 'CNRG', 'Deep_GCN_AE', 'Deep_GCN_VAE', \
          'Erdos-Renyi', 'GCN_AE', 'GCN_VAE', 'HRG', \
          'Linear_AE', 'Linear_VAE', 'NetGAN', 'SBM']

for model in models:
    print(f'starting {model} ... ')
    path = os.path.join(base_path, dataset, model)

    for subdir, dirs, files in os.walk(path):
        for filename in files:
            string = subdir.split('/')[-2:]
            file = os.path.join(subdir, filename)
            print(f'starting\t{string[-2]}\t{string[-1]}\t{filename} ... ')
            root = load_pickle(file)
            node = root
            while len(node.children) > 0:
                try:
                    node.robustness
                except AttributeError:
                    node = TreeNode(name=node.name, graph=node.graph, stats=node.stats, stats_seq=node.stats_seq, children=node.children)
                if node.robustness is None or node.robustness == {}:
                    graphstats = GraphStats(graph=node.graph, run_id=0)
                    graphstats._calculate_robustness_measures()
                    node.robustness = graphstats.stats
                    node = node.children[0]
            with open(file, 'wb') as f:
                pickle.dump(root, f)
            print('done')
