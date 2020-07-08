from collections import Counter
import os
import sys; sys.path.append('./../../')
import pickle
import numpy as np
import pandas as pd
import networkx as nx
import scipy.stats as st
import multiprocessing as mp
from pathlib import Path
from src.Tree import TreeNode
from src.utils import load_pickle
from src.graph_stats import GraphStats
from src.graph_comparison import GraphPairCompare

def load_data(base_path, bucket, dataset, model):
    path = os.path.join(base_path, bucket, dataset, model)
    for subdir, dirs, files in os.walk(path):
        for filename in files:
            if 'bucket3' not in subdir or 'HRG' in subdir:
                print(f'\tloading {subdir} {filename} ... ', end='', flush=True)
                pkl = load_pickle(os.path.join(subdir, filename))
                print('done')
                yield pkl, model

def mkdir_output(path):
    if not os.path.isdir(path):
        try:
            os.mkdir(path)
        except OSError:
            print('ERROR: could not make directory {path} for some reason')
    return

def unravel(root):
    if type(root) is list:
        return root
    else:
        graphs = [node.graph for node in [root] + list(root.descendants)]
        return graphs

def absolute(graphs):
    density0 = graphs[0].size()/graphs[0].order()
    for G in graphs[1:]:
        if G.order() == 0:
            density = np.nan
        else:
            density = G.size()/G.order() - density0
        yield density

def sequential(graphs):
    prev = graphs[0]
    for curr in graphs[1:]:
        if curr.size() == 0 or curr.order() == 0 or prev.size() == 0 or prev.order() == 0:
            density = np.nan
        else:
            density = curr.size()/curr.order() - prev.size()/prev.order()
        yield density
        prev = curr

def absolute_density(graphs):
    print('\t\tabsolute... ', end='', flush=True)
    abs_densities = [x for x in absolute(graphs)]
    print('done')
    return abs_densities

def sequential_density(graphs):
    print('\t\tsequential... ', end='', flush=True)
    seq_densities = [x for x in sequential(graphs)]
    print('done')
    return seq_densities

def length_chain(root):
    return len(root.descendants)

def flatten(L):
    return [item for sublist in L for item in sublist]

def compute_stats(densities):
    #padding = max(len(l) for l in js)
    #for idx, l in enumerate(js):
    #    while len(js[idx]) < padding:
    #        js[idx] += [np.NaN]
    for idx, l in enumerate(densities):
        if l == []:
            densities[idx] = [0]
    #print(densities)
    if densities == []:
        return [], []
    else:
        mean = np.nanmean(densities, axis=0)
    ci = []
    for row in np.asarray(densities).T:
        ci.append(st.t.interval(0.95, len(row)-1, loc=np.mean(row), scale=st.sem(row)))
    return np.asarray(mean), np.asarray(ci)

def construct_table(abs_densities, seq_densities, model):
    abs_mean, abs_ci = compute_stats(abs_densities)
    seq_mean, seq_ci = compute_stats(seq_densities)

    if abs_mean == []:
        rows = {'model': [], 'gen': [], 'abs_mean': [], 'abs-95%': [], 'abs+95%': [], 'seq_mean': [], 'seq-95%': [], 'seq+95%': []}
        return pd.DataFrame(rows)

    gen = [x + 1 for x in range(len(abs_mean))]

    rows = {'model': model, 'gen': gen, 'abs_mean': abs_mean, 'abs-95%': abs_ci[:,0], 'abs+95%': abs_ci[:,1], 'seq_mean': seq_mean, 'seq-95%': seq_ci[:,0], 'seq+95%': seq_ci[:,1]}

    df = pd.DataFrame(rows)
    return df

def main():
    base_path = '/data/infinity-mirror/buckets'
    dataset = 'chess'
    models = ['BTER', 'BUGGE', 'Chung-Lu', 'CNRG', 'Erdos-Renyi', 'HRG', 'SBM', 'NetGAN', 'GCN_AE', 'Linear_AE', 'GraphRNN', 'Kronecker']

    for model in models:
        for bucket in ['bucket1']:
            #output_path = os.path.join(base_path, dataset, models[0], 'jensen-shannon')
            output_path = '/home/dgonza26/infinity-mirror/data/density'
            mkdir_output(output_path)

            abs_densities = []
            seq_densities = []
            gen = []
            for root, model in load_data(base_path, bucket, dataset, model):
                graphs = unravel(root)
                assert graphs != []
                abs_densities.append(absolute_density(graphs))
                seq_densities.append(sequential_density(graphs))

            df = construct_table(abs_densities, seq_densities, model)
            df.to_csv(f'{output_path}/{dataset}_{model}_density.csv', float_format='%.7f', sep='\t', index=False, na_rep='nan')
            print(f'wrote: {output_path}/{dataset}_{model}_density.csv')

    return

main()
