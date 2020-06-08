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

base_path = '/data/infinity-mirror'
#base_path = '/Users/akira/data/'
dataset = 'flights'
models = ['GCN_AE', 'Linear_AE']

def get_graph_vec(g: nx.Graph, kernel: str='heat', dim: int=250, eigenvalues: int=20) -> np.ndarray:
    return net.netlsd(g, kernel=kernel, timescales=np.logspace(-2, 2, dim), eigenvalues=eigenvalues)

def compare_graphs(g1: nx.Graph, g2: nx.Graph, kernel: str='heat', dim: int=250, eigenvalues: int=20) -> float:
    g_vec1 = get_graph_vec(g=g1, kernel=kernel, dim=dim, eigenvalues=eigenvalues)
    g_vec2 = get_graph_vec(g=g2, kernel=kernel, dim=dim, eigenvalues=eigenvalues)
    return net.compare(g_vec1, g_vec2)

def get_row(root, cols, dataset, model, dim):
    if type(root) is list:
        for idx, graph in enumerate(root):
            row = {'name': dataset, 'level': idx, 'model': model}
            try:
                for i, x in enumerate(get_graph_vec(graph, dim=dim)):
                    row[f'v{i}'] = x
                yield row
            except ArpackNoConvergence as err:
                raise
    else:
        for tnode in [root] + list(root.descendants):
            row = {'name': dataset, 'level': tnode.depth, 'model': model}
            try:
                for i, x in enumerate(get_graph_vec(tnode.graph, dim=dim)):
                    row[f'v{i}'] = x
                yield row
            except ArpackNoConvergence as err:
                raise

def main():
    dim = 250
    cols = ['name', 'level', 'model'] + [f'v{i}' for i in range(dim)]
    save_path = os.path.join(base_path, 'stats', 'embedding')

    print(f'\nworking on {dataset}')
    for model in models:
        print(f'starting {model} ... ')
        path = os.path.join(base_path, dataset, model)

        rows = {col: [] for col in cols}
        for subdir, dirs, files in os.walk(path):
            if 'pagerank' not in subdir:
                for filename in files:
                    if '_augmented' not in filename and '_seq' not in filename and '_rob' not in filename and '.pkl.gz' in filename:
                        print(f'\ttrying {filename} ... ', end='', flush=True)
                        try:
                            root = load_pickle(os.path.join(subdir, filename))
                            for row in get_row(root, cols, dataset, model, dim):
                                for key, val in row.items():
                                    rows[key].append(val)
                            print('SUCCESS')
                        except ArpackNoConvergence:
                            print('FAILURE')
        df = pd.DataFrame(rows)
        df.to_csv(f'{save_path}/{dataset}_{model}.csv', index=False)

main()
