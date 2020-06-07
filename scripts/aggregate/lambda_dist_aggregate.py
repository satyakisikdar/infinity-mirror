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

def load_df(path):
    for subdir, dirs, files in os.walk(path):
        for filename in files:
            if 'lambda' in filename and 'csv' in filename:
                print(f'\tloading {subdir} {filename} ... ', end='', flush=True)
                df = pd.read_csv(os.path.join(subdir, filename), sep='\t'), filename
                print('done')
                yield df

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
    density0 = nx.density(graphs[0])
    for G in graphs[1:]:
        density = nx.density(G) - density0
        yield density

def sequential(graphs):
    prev = graphs[0]
    for curr in graphs[1:]:
        density = nx.density(curr) - nx.density(prev)
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
    print(densities)
    mean = np.nanmean(densities, axis=0)
    ci = []
    for row in np.asarray(densities).T:
        ci.append(st.t.interval(0.95, len(row)-1, loc=np.mean(row), scale=st.sem(row)))
    return np.asarray(mean), np.asarray(ci)

def construct_table(abs_densities, seq_densities, model):
    abs_mean, abs_ci = compute_stats(abs_densities)
    seq_mean, seq_ci = compute_stats(seq_densities)
    gen = [x + 1 for x in range(len(abs_mean))]

    rows = {'model': model, 'gen': gen, 'abs_mean': abs_mean, 'abs-95%': abs_ci[:,0], 'abs+95%': abs_ci[:,1], 'seq_mean': seq_mean, 'seq-95%': seq_ci[:,0], 'seq+95%': seq_ci[:,1]}

    df = pd.DataFrame(rows)
    return df

def abs_mean(a):
    return np.mean(a)

def abs95u(a):
    return st.t.interval(0.95, len(a) - 1, loc=np.mean(a), scale=st.sem(a))[1]

def abs95d(a):
    return st.t.interval(0.95, len(a) - 1, loc=np.mean(a), scale=st.sem(a))[0]

def main():
    base_path = '/data/infinity-mirror/stats/lambda'
    output_path = '/home/dgonza26/infinity-mirror/data/lambda'

    for df, filename in load_df(base_path):
        if 'seq' in df.columns:
            df = df.drop(['trial', 'seq'], axis=1).groupby(['model', 'gen']).agg([abs_mean, abs95d, abs95u])
        else:
            df = df.drop('trial', axis=1).groupby(['model', 'gen']).agg([abs_mean, abs95d, abs95u])
        df.columns = df.columns.droplevel(0)
        for column in df:
            if column != 'gen' and column != 'trial' and column != 'model':
                df[column] = df[column].apply(lambda x: 0.001 if x < 0.001 else x)
        df.to_csv(f'{output_path}/{filename}', float_format='%.7f', sep='\t', na_rep='nan')
        print(f'wrote: {output_path}/{filename}')

    #df.to_csv(f'{output_path}/{dataset}_{model}_density.csv', float_format='%.7f', sep='\t', index=False, na_rep='nan')

    return

main()
