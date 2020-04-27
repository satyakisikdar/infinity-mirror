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
dataset = 'tree'
#model = 'BTER'
models = ['BTER', 'BUGGE', 'Chung-Lu', 'CNRG', 'Erdos-Renyi', 'HRG', 'NetGAN', 'SBM']
#models = ['BTER', 'BUGGE', 'Chung-Lu', 'CNRG', 'Erdos-Renyi', 'HRG', 'NetGAN', 'SBM', \
#          'Kronecker', 'Deep_GCN_AE', 'Deep_GCN_VAE', 'GCN_AE', 'GCN_VAE', 'Linear_AE', 'Linear_VAE']

for model in models:
    print(f'starting {model} ... ')
    path = os.path.join(base_path, dataset, model)

    for subdir, dirs, files in os.walk(path):
        for filename in files:
            string = subdir.split('/')[-2:]
            file = os.path.join(subdir, filename)
            newfile = file[:-7] + '_augmented.pkl.gz'
            print(f'starting\t{string[-2]}\t{string[-1]}\t{filename} ... ')
            root = load_pickle(file)

            try:
                root.robustness
            except AttributeError:
                root = TreeNode(name=root.name, graph=root.graph, stats=root.stats, stats_seq=root.stats_seq, children=root.children)
            if root.robustness is None or root.robustness == {}:
                graphstats = GraphStats(graph=root.graph, run_id=0)
                graphstats._calculate_robustness_measures()
                root.robustness = graphstats.stats

            for node in root.descendants:
                try:
                    node.robustness
                except AttributeError:
                    node = TreeNode(name=node.name, graph=node.graph, stats=node.stats, stats_seq=node.stats_seq, children=node.children)
                if node.robustness is None or node.robustness == {}:
                    graphstats = GraphStats(graph=node.graph, run_id=0)
                    graphstats._calculate_robustness_measures()
                    node.robustness = graphstats.stats
            with open(newfile, 'wb') as f:
                pickle.dump(root, f)
            print('done')
