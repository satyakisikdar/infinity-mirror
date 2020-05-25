import os
import sys; sys.path.append('./..')
import pickle
import numpy as np
import networkx as nx # requires 2.3.0
import pandas as pd; pd.options.display.float_format = '{:,.5f}'.format
import statsmodels.stats.api as sm
import warnings; warnings.filterwarnings("ignore", category=UserWarning)
from glob import glob
from statistics import median_low
from scipy.sparse.linalg.eigen.arpack import eigs, ArpackNoConvergence

from src.Tree import TreeNode
from src.utils import load_pickle
from src.graph_stats import GraphStats
from src.graph_comparison import GraphPairCompare

graphs = ['eucore', 'clique-ring-500-4', 'flights', 'tree', 'chess']

base_path = '/data/dgonza26'
dataset = 'eucore'
models = ['Linear_VAE']

def main():
    print(f'\nworking on {dataset}')
    for model in models:
        print(f'starting {model} ... ')
        path = os.path.join(base_path, dataset, model)
        for subdir, dirs, files in os.walk(path):
            for filename in files:
                #if '_augmented' not in filename:
                if True:
                    print(f'\tchecking {filename}')
                    root = load_pickle(os.path.join(subdir, filename))
                    root = root.children[0]
                    root = root.children[0]
                    try:
                        print(f'\tstats_seq: {root.stats_seq}')
                        print('\tSUCCESS: stats_seq')
                    except AttributeError:
                        print('\tFAILURE: stats_seq')
                    try:
                        print(f'\trobustness: {root.robustness}')
                        print('\tSUCCESS: robustness')
                    except AttributeError:
                        print('\tFAILURE: robustness')
                    return

main()
