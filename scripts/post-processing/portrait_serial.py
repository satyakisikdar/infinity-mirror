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
# import src.netrd as netrd
from src.portrait.portrait_divergence import portrait_divergence

def load_data(base_path, dataset, model, seq_flag, rob_flag):
    path = os.path.join(base_path, dataset, model)
    for subdir, dirs, files in os.walk(path):
        for filename in files:
            if 'csv' not in filename:
                # if 'seq' not in filename and 'rob' not in filename:
                print(f'loading {subdir} {filename} ... ', end='', flush=True)
                pkl = load_pickle(os.path.join(subdir, filename))
                trial = filename.split('_')[2].strip('.pkl.gz')
                print('done')
                yield pkl, trial


def mkdir_output(path):
    if not os.path.isdir(path):
        try:
            os.mkdir(path)
        except OSError:
            print('ERROR: could not make directory {path} for some reason')
    return


def compute_graph_stats(root):
    print('computing GraphStats... ', end='', flush=True)
    if type(root) is list:
        graph_stats = [GraphStats(graph=g, run_id = 1) for g in root]
    else:
        graph_stats = [GraphStats(graph=node.graph, run_id=1) for node in [root] + list(root.descendants)]
    print('done')
    return graph_stats


def absolute(graph_stats):
    for gs in graph_stats[1:]:
        comparator = GraphPairCompare(graph_stats[0], gs)
        delta = comparator.deltacon0()
        yield delta


def sequential(graph_stats):
    prev = graph_stats[0]
    for curr in graph_stats[1:]:
        comparator = GraphPairCompare(prev, curr)
        delta = comparator.deltacon0()
        yield delta
        prev = curr


def absolute_delta(graph_stats):
    print('absolute... ', end='', flush=True)
    abs_delta = [x for x in absolute(graph_stats)]
    print('done')
    return abs_delta


def sequential_delta(graph_stats):
    print('sequential... ', end='', flush=True)
    seq_delta = [x for x in sequential(graph_stats)]
    print('done')
    return seq_delta


def length_chain(root):
    return len(root.descendants)


def flatten(L):
    return [item for sublist in L for item in sublist]


def compute_stats(delta):
    padding = max(len(l) for l in delta)
    for idx, l in enumerate(delta):
        while len(delta[idx]) < padding:
            delta[idx] += [np.NaN]
    mean = np.nanmean(delta, axis=0)
    ci = []
    for row in np.asarray(delta).T:
        ci.append(st.t.interval(0.95, len(row)-1, loc=np.mean(row), scale=st.sem(row)))
    return np.asarray(mean), np.asarray(ci)


def construct_table(abs_delta, seq_delta, model):
    abs_mean, abs_ci = compute_stats(abs_delta)
    seq_mean, seq_ci = compute_stats(seq_delta)
    gen = [x + 1 for x in range(len(abs_mean))]

    rows = {'model': model, 'gen': gen, 'abs_mean': abs_mean, 'abs-95%': abs_ci[:,0], 'abs+95%': abs_ci[:,1], 'seq_mean': seq_mean, 'seq-95%': seq_ci[:,0], 'seq+95%': seq_ci[:,1]}

    df = pd.DataFrame(rows)
    return df


def construct_full_table(abs_delta, seq_delta, model, trials):
    gen = []
    for t in trials:
        gen += [x + 1 for x in range(len(t))]

    rows = {'model': model, 'trial': flatten(trials), 'gen': gen, 'abs': abs_delta, 'seq': seq_delta}

    df = pd.DataFrame(rows)
    return df


if __name__ == '__main__':
    base_path = '/data/infinity-mirror/'
    dataset = 'clique-ring-500-4'
    models = ['Chung-Lu', 'CNRG', 'SBM']
    # models = ['BTER', 'BUGGE', 'CNRG', 'Chung-Lu', 'Erdos-Renyi', 'SBM', 'HRG', 'NetGAN']

    output_path = os.path.join(base_path, 'stats/portrait/')
    mkdir_output(output_path)

    cols = ['model', 'gen', 'trial_id', 'portrait']

    for model in models:
        abs_delta = []
        trials = []
        rows = {col: [] for col in cols}
        for root, trial in load_data(base_path, dataset, model, True, False):  # add TQDM
            if isinstance(root, list):
                root_graph = root[0]
                descendants = root[1: ]
            else:
                root_graph = root.graph
                descendants = [tnode.graph for tnode in root.descendants]

            for i, desc_graph in enumerate(descendants, 1):
                d = portrait_divergence(G=root_graph, H=desc_graph)

                rows['model'].append(model)
                rows['gen'].append(i)
                rows['trial_id'].append(int(trial))
                rows['portrait'].append(d)
            break
        df = pd.DataFrame(rows)
        df.to_csv(f'{output_path}/{dataset}_{model}_portrait.csv', float_format='%.7f', sep='\t', index=False, na_rep='nan')
