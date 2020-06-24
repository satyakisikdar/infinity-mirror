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
    for model in models:
        path = os.path.join(base_path, dataset, model)
        for subdir, dirs, files in os.walk(path):
            for filename in files:
                if 'csv' not in filename:
                    if 'seq' not in filename and 'rob' not in filename:
                        print(f'loading {subdir} {filename} ... ', end='', flush=True)
                        pkl = load_pickle(os.path.join(subdir, filename))#, subdir.split('/')[-1]
                        print('done')
                        yield pkl, filename

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

def compute_pagerank(graph_stats):
    print('computing pagerank... ', end='', flush=True)
    pgds = [gs.pagerank() for gs in graph_stats]
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

def construct_full_table(pageranks, trials, gens, model):
    cols = []
    for pagerank in pageranks:
        cols.append(pagerank)

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

def main():
    base_path = '/data/infinity-mirror'
    dataset = 'clique-ring-500-4'
    models = ['GCN_AE', 'Linear_AE']
    model = models[0]

    output_path = os.path.join(base_path, dataset, models[0], 'pagerank')
    mkdir_output(output_path)

    pageranks = []
    for root, filename in load_data(base_path, dataset, models, True, False):
        graph_stats = compute_graph_stats(root)
        pagerank = compute_pagerank(graph_stats)
        for idx, pr in enumerate(pagerank):
            df = pd.DataFrame({'node': list(pr.keys()), 'pagerank': list(pr.values())})
            df.to_csv(f'{output_path}/{filename}_pagerank_{idx}.csv', float_format='%.7f', sep='\t', index=False, na_rep='nan')

    return

main()
