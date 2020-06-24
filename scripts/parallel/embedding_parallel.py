import os
import sys;
from os import listdir
from os.path import isfile, join

from tqdm import tqdm

sys.path.append('./..')
sys.path.append('./../../')
import numpy as np
import NetLSD.netlsd as net
import networkx as nx # requires 2.3.0
import pandas as pd; pd.options.display.float_format = '{:,.5f}'.format
import warnings; warnings.filterwarnings("ignore", category=UserWarning)
from scipy.sparse.linalg.eigen.arpack import ArpackNoConvergence

from src.utils import load_pickle, ColorPrint, verify_dir, verify_file
import multiprocessing as mp


def get_graph_vec(g: nx.Graph, kernel: str='heat', dim: int=250, eigenvalues: int=20) -> np.ndarray:
    return net.netlsd(g, kernel=kernel, timescales=np.logspace(-2, 2, dim), eigenvalues=eigenvalues)

def compare_graphs(g1: nx.Graph, g2: nx.Graph, kernel: str='heat', dim: int=250, eigenvalues: int=20) -> float:
    g_vec1 = get_graph_vec(g=g1, kernel=kernel, dim=dim, eigenvalues=eigenvalues)
    g_vec2 = get_graph_vec(g=g2, kernel=kernel, dim=dim, eigenvalues=eigenvalues)
    return net.compare(g_vec1, g_vec2)

#todo: what does this function do that the one in graph embedding doesn't?
def get_row(root, cols, dataset, model, trial, dim):
    if type(root) is list:
        for idx, graph in enumerate(root):
            row = {'name': dataset, 'level': idx, 'model': model, 'trial': trial}
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

def load_data(input_path, dataset, model):
    path = os.path.join(input_path, dataset, model)
    input_filenames = [f for f in listdir(path) if isfile(join(path, f))]
    # print(input_filenames)
    for filename in input_filenames:
        pkl = load_pickle(os.path.join(path, filename))
        trial = filename.split('_')[2].strip('.pkl.gz')
        yield pkl, trial

def parallel_computation(input_path, dataset, model):
    output_path = os.path.join('/data/infinity-mirror/', 'output', 'embedding/' ,model)
    verify_dir(output_path)

    filename = f'{output_path}/embedding_{dataset}_{model}.csv'
    if verify_file(filename):
        df = pd.read_csv(filename, sep='\t')
        return df, dataset, model

    rows = {col: [] for col in cols}

    for graph_list, trial in tqdm(load_data(input_path, dataset, model)):
        temp_filename = f'{output_path}/embedding_{dataset}_{model}_temp_{trial}.csv'
        try:
            for row in get_row(graph_list, cols, dataset, model, trial, dim):
                for key, val in row.items():
                    rows[key].append(val)
            # print('SUCCESS')
        except ArpackNoConvergence:
            print('FAILURE')
        inc_df = pd.DataFrame(rows)
        inc_df.to_csv(temp_filename, float_format='%.7f', sep='\t', index=False, na_rep='nan')

    df = pd.DataFrame(rows)    # save out incremental work in case something goes wrong
    df.to_csv(filename, float_format='%.7f', sep='\t', index=False, na_rep='nan')
    return df, dataset, model

if __name__ == '__main__':
    #dac
    # input_path = '/afs/crc.nd.edu/user/t/tford5/infinity-mirror/cleaned/'
    #DSGs
    input_path = '/data/infinity-mirror/cleaned/'

    output_path = os.path.join('/data/infinity-mirror/', 'output', 'embedding/')
    verify_dir(output_path)
    datasets = ['eucore','tree', 'flights', 'clique-ring-500-4']
    models = ['BTER', 'BUGGE', 'CNRG', 'Chung-Lu', 'Erdos-Renyi', 'HRG', 'SBM', 'NetGAN', 'GCN_AE', 'Linear_AE']
    #test
    # datasets = ['eucore']
    # models = ['CNRG']

    dim = 250
    cols = ['name', 'level', 'model', 'trial'] + [f'v{i}' for i in range(dim)]

    parallel_args = []
    results = []

    for dataset in datasets:
        for model in models:
            parallel_args.append([input_path, dataset, model])

    pbar = tqdm(len(parallel_args))

    def result_update(result):
        pbar.update()
        pbar.set_postfix_str(result[1]+result[2])
        results.append(result[0])

    # sequential implementation
    # for p_arg in parallel_args:
    #     results.append(parallel_computation(p_arg[0],p_arg[1],p_arg[2]))

    asyncResults = []
    n_threads = 4
    with mp.Pool(n_threads) as outerPool:
        ColorPrint.print_green(f"Starting Pool with {n_threads} threads with {len(parallel_args)} tasks.")
        for p_arg in parallel_args:
            r = outerPool.apply_async(parallel_computation, p_arg, callback=result_update)
            asyncResults.append(r)
        for r in asyncResults:
            try:
                r.wait()
            except:
                continue

        df = pd.concat(results)
        df.to_csv(f'{output_path}/{dataset}_{model}.csv', index=False)