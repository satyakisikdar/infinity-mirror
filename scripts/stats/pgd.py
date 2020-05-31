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

def load_data(base_path, dataset, models, seq_flag, rob_flag):
    count = 0
    for model in models:
        path = os.path.join(base_path, dataset, model)
        for subdir, dirs, files in os.walk(path):
            for filename in files:
                if 'jensen-shannon' not in subdir and 'lambda-dist' not in subdir:
                    if (seq_flag or 'seq' not in filename) and (rob_flag or 'rob' not in filename):
                        count += 1
                        if count > 3:
                            break
                        print(f'loading {subdir} {filename} ... ', end='', flush=True)
                        pkl = load_pickle(os.path.join(subdir, filename))#, subdir.split('/')[-1]
                        print('done')
                        yield pkl, filename.split('_')[2]

def mkdir_output(path):
    if not os.path.isdir(path):
        try:
            os.mkdir(path)
        except OSError:
            print('ERROR: could not make directory {path} for some reason')
    return

def compute_graph_stats(root):
    print('computing GraphStats... ', end='', flush=True)
    graph_stats = [GraphStats(graph=node.graph, run_id=1) for node in [root] + list(root.descendants)]
    print('done')
    return graph_stats

def compute_pgd(graph_stats):
    print(graph_stats[0].pgd_graphlet_counts())
    pgds = [gs.pgd_graphlet_counts() for gs in graph_stats]
    return pgds

def length_chain(root):
    return len(root.descendants)

def flatten(L):
    return [item for sublist in L for item in sublist]

def compute_stats(ld):
    padding = max(len(l) for l in ld)
    for idx, l in enumerate(ld):
        while len(ld[idx]) < padding:
            ld[idx] += [np.NaN]
    mean = np.nanmean(ld, axis=0)
    ci = []
    for row in np.asarray(ld).T:
        ci.append(st.t.interval(0.95, len(row)-1, loc=np.mean(row), scale=st.sem(row)))
    return np.asarray(mean), np.asarray(ci)

def construct_agregate_table(abs_ld, seq_ld, model):
    abs_mean, abs_ci = compute_stats(abs_ld)
    seq_mean, seq_ci = compute_stats(seq_ld)
    gen = [x + 1 for x in range(len(abs_mean))]

    rows = {'model': model, 'gen': gen, 'abs_mean': abs_mean, 'abs-95%': abs_ci[:,0], 'abs+95%': abs_ci[:,1], 'seq_mean': seq_mean, 'seq-95%': seq_ci[:,0], 'seq+95%': seq_ci[:,1]}

    df = pd.DataFrame(rows)
    return df

def construct_full_table(abs_ld, seq_ld, trials, gens, model):
    #abs_mean, abs_ci = compute_stats(abs_ld)
    #seq_mean, seq_ci = compute_stats(seq_ld)
    #gen = [x + 1 for x in range(len(abs_mean))]

    print(gens)
    print(trials)
    rows = {'model': model, 'gen': gens, 'trial': trials, 'abs': abs_ld, 'seq': seq_ld}

    df = pd.DataFrame(rows)
    return df

def main():
    base_path = '/data/infinity-mirror'
    dataset = 'eucore'
    models = ['BTER']
    model = models[0]

    output_path = os.path.join(base_path, dataset, models[0], 'lambda-dist')
    mkdir_output(output_path)

    pgds = []
    trials = []
    gens = []
    for root, trial in load_data(base_path, dataset, models, True, False):
        graph_stats = compute_graph_stats(root)
        pgds += compute_pgd(graph_stats)
        trials += [trial for _ in graph_stats]
        gens.append(x + 1 for x in range(len(graph_stats)))

    print(pgds)
    df_agregate = construct_agregate_table(abs_ld, seq_ld, model)
    df_agregate.to_csv(f'{output_path}/{dataset}_{model}_pgd_ag.csv', float_format='%.7f', sep='\t', index=False, na_rep='nan')
    df_full = construct_full_table(abs_ld, seq_ld, trials, gens, model)
    df_full.to_csv(f'{output_path}/{dataset}_{model}_pgd_full.csv', float_format='%.7f', sep='\t', index=False, na_rep='nan')

    return

main()
