from collections import Counter
import os
import sys; sys.path.append('./')
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
    for model in models:
        path = os.path.join(base_path, dataset, model)
        for subdir, dirs, files in os.walk(path):
            for filename in files:
                if 'csv' not in filename:
                    if (seq_flag or 'seq' not in filename) and (rob_flag or 'rob' not in filename):
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
    if type(root) is list:
        graph_stats = [GraphStats(graph=g, run_id=1) for g in root]
    else:
        graph_stats = [GraphStats(graph=node.graph, run_id=1) for node in [root] + list(root.descendants)]
    print('done')
    return graph_stats

def compute_pgd(graph_stats):
    print('computing pgd... ', end='', flush=True)
    pgds = [gs.pgd_graphlet_counts() for gs in graph_stats]
    print('done')
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

def construct_full_table(pgds, trials, gens, model):
    total_2_1edge = []
    total_2_indep = []
    total_3_tris = []
    total_2_star = []
    total_3_1edge = []
    total_4_clique = []
    total_4_chordcycle = []
    total_4_tailed_tris = []
    total_3_star = []
    total_4_path = []
    total_4_1edge = []
    total_4_2edge = []
    total_4_2star = []
    total_4_tri = []
    total_4_indep = []
    for d in pgds:
        total_2_1edge.append(d['total_2_1edge'])
        total_2_indep.append(d['total_2_indep'])
        total_3_tris.append(d['total_3_tris'])
        total_2_star.append(d['total_2_star'])
        total_3_1edge.append(d['total_3_1edge'])
        total_4_clique.append(d['total_4_clique'])
        total_4_chordcycle.append(d['total_4_chordcycle'])
        total_4_tailed_tris.append(d['total_4_tailed_tris'])
        total_3_star.append(d['total_3_star'])
        total_4_path.append(d['total_4_path'])
        total_4_1edge.append(d['total_4_1edge'])
        total_4_2edge.append(d['total_4_2edge'])
        total_4_2star.append(d['total_4_2star'])
        total_4_tri.append(d['total_4_tri'])
        total_4_indep.append(d['total_4_indep'])

    rows = {'model': model, 'gen': gens, 'trial': trials, \
            'total_2_1edge': total_2_1edge, \
            'total_2_indep': total_2_indep, \
            'total_3_tris': total_3_tris, \
            'total_2_star': total_2_star, \
            'total_3_1edge': total_3_1edge, \
            'total_4_clique': total_4_clique, \
            'total_4_chordcycle': total_4_chordcycle, \
            'total_4_tailed_tris': total_4_tailed_tris, \
            'total_3_star': total_3_star, \
            'total_4_path': total_4_path, \
            'total_4_1edge': total_4_1edge, \
            'total_4_2edge': total_4_2edge, \
            'total_4_2star': total_4_2star, \
            'total_4_tri': total_4_tri, \
            'total_4_indep': total_4_indep
            }

    df = pd.DataFrame(rows)
    return df

if __name__ == '__main__':
    base_path = '/data/infinity-mirror'
    dataset = 'eucore'
    models = ['Kronecker']
    model = models[0]

    output_path = os.path.join('/data/infinity-mirror/stats', 'pgd')
    mkdir_output(output_path)

    pgds = []
    trials = []
    gens = []
    for root, trial in load_data(base_path, dataset, models, True, False):
        graph_stats = compute_graph_stats(root)
        pgds += compute_pgd(graph_stats)
        trials += [trial for _ in graph_stats]
        gens += [x for x in range(len(graph_stats))]

    df_full = construct_full_table(pgds, trials, gens, model)
    df_full.to_csv(f'{output_path}/{dataset}_{model}_pgd_full.csv', float_format='%.7f', sep='\t', index=False, na_rep='nan')

