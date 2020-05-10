import os
import sys; sys.path.append('./..')
import pickle
import numpy as np
import netlsd as net
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

#base_path = '/data/dgonza26'
base_path = '/Users/akira/data/'
dataset = 'clique-ring-500-4'
models = ['BTER', 'BUGGE', 'CNRG', 'Chung-Lu', 'Erdos-Renyi', 'HRG', 'Kronecker', 'NetGAN', 'SBM']

def get_graph_vec(g: nx.Graph, kernel: str='heat', dim: int=250, eigenvalues: int=20) -> np.ndarray:
    return net.netlsd(g, kernel=kernel, timescales=np.logspace(-2, 2, dim), eigenvalues=eigenvalues)

def compare_graphs(g1: nx.Graph, g2: nx.Graph, kernel: str='heat', dim: int=250, eigenvalues: int=20) -> float:
    g_vec1 = get_graph_vec(g=g1, kernel=kernel, dim=dim, eigenvalues=eigenvalues)
    g_vec2 = get_graph_vec(g=g2, kernel=kernel, dim=dim, eigenvalues=eigenvalues)
    return net.compare(g_vec1, g_vec2)

def get_row(root, cols, dataset, model, dim):
    for tnode in [root] + list(root.descendants):
        row = {'name': dataset, 'level': tnode.depth, 'model': model}
        for i, x in enumerate(get_graph_vec(tnode.graph, dim=dim)):
            row[f'v{i}'] = x
        yield row

def main():
    dim = 250
    cols = ['name', 'level', 'model'] + [f'v{i}' for i in range(dim)]
    rows = {col: [] for col in cols}
    save_path = os.path.join(base_path, dataset)

    for model in models:
        print(f'\nstarting {model} ... ')
        path = os.path.join(base_path, dataset, model)

        for subdir, dirs, files in os.walk(path):
            for filename in files:
                if 'augmented' not in filename and '.pkl.gz' in filename:
                    print(f'\ttrying {filename} ... ', end='', flush=True)
                    try:
                        root = load_pickle(os.path.join(subdir, filename))
                        trows = {col: [] for col in cols}
                        for row in get_row(root, cols, dataset, model, dim):
                            for key, val in row.items():
                                trows[key].append(val)
                        for key, val in trows.items():
                            rows[key].append(val)
                        print('SUCCESS')
                    except ArpackNoConvergence:
                        print('FAILURE')

    df = pd.DataFrame(rows)
    df.to_csv(f'{save_path}/{dataset}.csv')

main()
