from collections import Counter
import os
import sys; sys.path.append('./../../')
import pickle
import numpy as np
import pandas as pd
import networkx as nx
import scipy.stats as st
import multiprocessing as mp
from src.Tree import TreeNode
from src.utils import load_pickle
from src.graph_stats import GraphStats
from src.graph_comparison import GraphPairCompare

def load_data(base_path, dataset, models, flag):
    for model in models:
        path = os.path.join(base_path, dataset, model)
        for subdir, dirs, files in os.walk(path):
            for filename in files:
                if 'jensen-shannon' not in subdir:
                    if (flag or 'seq' not in filename) and 'rob' not in filename:
                        print(f'loading {subdir} {filename}')
                        yield load_pickle(os.path.join(subdir, filename)), subdir.split('/')[-1]

def compute_graph_stats(root):
    return [GraphStats(graph=node.graph, run_id=1) for node in [root] + list(root.descendants)]

def absolute(graph_stats):
    for gs in graph_stats[1:]:
        comparator = GraphPairCompare(graph_stats[0], gs)
        dist = comparator.js_distance()
        yield dist

def sequential(graph_stats):
    prev = graph_stats[0]
    for curr in graph_stats[1:]:
        comparator = GraphPairCompare(prev, curr)
        dist = comparator.js_distance()
        yield dist
        prev = curr

def absolute_js(graph_stats):
    print('absolute... ', end='', flush=True)
    abs_js = [x for x in absolute(graph_stats)]
    print('done')
    return abs_js

def sequential_js(graph_stats):
    print('sequential... ', end='', flush=True)
    seq_js = [x for x in sequential(graph_stats)]
    print('done')
    return seq_js

def length_chain(root):
    return len(root.descendants)

def flatten(L):
    return [item for sublist in L for item in sublist]

def compute_stats(js):
    padding = max(len(l) for l in js)
    for idx, l in enumerate(js):
        while len(js[idx]) < padding:
            js[idx] += [np.NaN]
    mean = np.nanmean(js, axis=0)
    ci = []
    for row in np.asarray(js).T:
        ci.append(st.t.interval(0.95, len(row)-1, loc=np.mean(row), scale=st.sem(row)))
    return np.asarray(mean), np.asarray(ci)

def construct_table(abs_js, seq_js, model):
    abs_mean, abs_ci = compute_stats(abs_js)
    seq_mean, seq_ci = compute_stats(seq_js)
    gen = [x + 1 for x in range(len(abs_mean))]

    rows = {'model': model, 'gen': gen, 'abs_mean': abs_mean, 'abs-95%': abs_ci[:,0], 'abs+95%': abs_ci[:,1], 'seq_mean': seq_mean, 'seq-95%': seq_ci[:,0], 'seq+95%': seq_ci[:,1]}

    df = pd.DataFrame(rows)
    return df

def main():
    base_path = '/data/infinity-mirror'
    dataset = 'eucore'
    models = ['BTER']
    model = models[0]

    output_path = os.path.join(base_path, dataset, models[0], 'jensen-shannon')

    abs_js = []
    seq_js = []
    gen = []
    for root, model in load_data(base_path, dataset, models, True):
        graph_stats = compute_graph_stats(root)
        abs_js.append(absolute_js(graph_stats))
        seq_js.append(sequential_js(graph_stats))

    df = construct_table(abs_js, seq_js, model)
    df.to_csv(f'{output_path}/{dataset}_{model}', float_format='%.7f', sep='\t', index=False, na_rep='nan')

    return

main()
