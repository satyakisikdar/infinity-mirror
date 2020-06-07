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

def load_data(base_path, dataset, model, seq_flag, rob_flag):
    path = os.path.join(base_path, dataset, model)
    for subdir, dirs, files in os.walk(path):
        for filename in files:
            if 'csv' not in filename:
                if 'seq' not in filename and 'rob' not in filename:
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
        lambda_dist = comparator.lambda_dist()
        yield lambda_dist

def sequential(graph_stats):
    prev = graph_stats[0]
    for curr in graph_stats[1:]:
        comparator = GraphPairCompare(prev, curr)
        lambda_dist = comparator.lambda_dist()
        yield lambda_dist
        prev = curr

def absolute_lambda(graph_stats):
    print('absolute... ', end='', flush=True)
    abs_lambda = [x for x in absolute(graph_stats)]
    print('done')
    return abs_lambda

def sequential_lambda(graph_stats):
    print('sequential... ', end='', flush=True)
    seq_lambda = [x for x in sequential(graph_stats)]
    print('done')
    return seq_lambda

def length_chain(root):
    return len(root.descendants)

def flatten(L):
    return [item for sublist in L for item in sublist]

def compute_stats(lambda_dist):
    padding = max(len(l) for l in lambda_dist)
    for idx, l in enumerate(lambda_dist):
        while len(lambda_dist[idx]) < padding:
            lambda_dist[idx] += [np.NaN]
    mean = np.nanmean(lambda_dist, axis=0)
    ci = []
    for row in np.asarray(lambda_dist).T:
        ci.append(st.t.interval(0.95, len(row)-1, loc=np.mean(row), scale=st.sem(row)))
    return np.asarray(mean), np.asarray(ci)

def construct_table(abs_lambda, seq_lambda, model):
    abs_mean, abs_ci = compute_stats(abs_lambda)
    seq_mean, seq_ci = compute_stats(seq_lambda)
    gen = [x + 1 for x in range(len(abs_mean))]

    rows = {'model': model, 'gen': gen, 'abs_mean': abs_mean, 'abs-95%': abs_ci[:,0], 'abs+95%': abs_ci[:,1], 'seq_mean': seq_mean, 'seq-95%': seq_ci[:,0], 'seq+95%': seq_ci[:,1]}

    df = pd.DataFrame(rows)
    return df

def construct_full_table(abs_lambda, seq_lambda, model, trials):
    #abs_mean, abs_ci = compute_stats(abs_lambda)
    #seq_mean, seq_ci = compute_stats(seq_lambda)
    #gen = [x + 1 for x in range(len(abs_mean))]

    gen = []
    for t in trials:
        gen += [x + 1 for x in range(len(t))]

    rows = {'model': model, 'trial': flatten(trials), 'gen': gen, 'abs': abs_lambda}#, 'seq': seq_lambda}

    df = pd.DataFrame(rows)
    return df

def main():
    base_path = '/data/infinity-mirror'
    input_path = '/home/dgonza26/infinity-mirror/input'
    dataset = 'chess'
    models = ['HRG', 'NetGAN']

    output_path = os.path.join(base_path, 'stats', 'lambda')
    mkdir_output(output_path)

    for model in models:
        abs_lambda = []
        seq_lambda = []
        trials = []
        for root, trial in load_data(base_path, dataset, model, True, False):
            graph_stats = compute_graph_stats(root)
            trials.append([trial for _ in graph_stats[1:]])
            try:
                assert root.children[0].stats['lambda_dist'] is not None
                assert root.children[0].stats['lambda_dist'] != {}
            except Exception as e:
                #abs_lambda.append(absolute_lambda(graph_stats))
                #seq_lambda.append(sequential_lambda(graph_stats))
                abs_lambda += absolute_lambda(graph_stats)
            else:
                abs_lambda += [node.stats['lambda_dist'] for node in root.descendants]
            #try:
            #    assert root.children[0].stats_seq['lambda_dist'] is not None
            #except Exception as e:
            #    seq_lambda += sequential_lambda(graph_stats)
            #else:
            #    seq_lambda += [node.stats_seq['lambda_dist'] for node in root.descendants]

        if abs_lambda == []:
            print(f'SOMETHING WENT WRONG WITH {model}')
            exit()

        df = construct_full_table(abs_lambda, seq_lambda, model, trials)
        df.to_csv(f'{output_path}/{dataset}_{model}_lambda.csv', float_format='%.7f', sep='\t', index=False, na_rep='nan')
        print(f'wrote {output_path}/{dataset}_{model}_lambda.csv')
        #df.to_csv(f'{output_path}/{dataset}_{model}_lambda.csv', float_format='%.7f', sep='\t', index=False, na_rep='nan')

    return

main()
