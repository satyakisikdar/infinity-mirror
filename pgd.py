from collections import Counter
import os
import sys;
from os import listdir
from os.path import isfile, join

from tqdm import tqdm

sys.path.append('./')
import pickle
import numpy as np
import pandas as pd
import networkx as nx
import scipy.stats as st
import multiprocessing as mp
from src.Tree import TreeNode
from src.utils import load_pickle, ColorPrint, verify_file
from src.graph_stats import GraphStats
from src.graph_comparison import GraphPairCompare

def load_data(input_path, dataset, model, filename_idx):
    path = os.path.join(input_path, dataset, model)
    input_filenames = [f for f in listdir(path) if isfile(join(path, f))]
    # print(input_filenames)
    filename = input_filenames[filename_idx]
    pkl = load_pickle(os.path.join(path, filename))
    trial = filename.split('_')[2].strip('.pkl.gz')
    return pkl, trial

def get_trial_id(input_path, dataset, model, filename_idx):
    path = os.path.join(input_path, dataset, model)
    input_filenames = [f for f in listdir(path) if isfile(join(path, f))]
    filename = input_filenames[filename_idx]
    trial = filename.split('_')[2].strip('.pkl.gz')
    return trial

def mkdir_output(path):
    if not os.path.isdir(path):
        try:
            os.mkdir(path)
        except OSError:
            print(f'ERROR: could not make directory {path} for some reason')
    return

def compute_graph_stats(root):
    print('computing GraphStats... ', end='', flush=True)
    if type(root) is list:
        graph_stats = [GraphStats(graph=g, run_id=1) for g in root]
    else:
        graph_stats = [GraphStats(graph=node.graph, run_id=1) for node in [root] + list(root.descendants)]
    print('done')
    return graph_stats

def compute_pgd(graph_stats, n_threads=4):
    print('computing pgd... ', end='', flush=True)
    pgds = [gs.pgd_graphlet_counts(n_threads=n_threads) for gs in graph_stats]
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
    total_4_cycle = []
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
        total_4_cycle.append(d['total_4_cycle'])
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
            'total_4_cycle': total_4_cycle, \
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

def sublevel_parallel_computation(input_path, dataset, model, idx):
    output_path = f'/afs/crc.nd.edu/user/t/tford5/infinity-mirror/output/pgd/{model}/'
    mkdir_output(output_path)
    trial = get_trial_id(input_path, dataset, model, idx)

    output_filename = f'{output_path}/{dataset}_{model}_{trial}_pgd_full.csv'

    # don't do repeat work
    if verify_file(output_filename):
        ColorPrint.print_orange(f'{output_filename} Already Exists!')
        return dataset+model

    graph_list, trial = load_data(input_path, dataset, model, idx)

    pgds = []
    trials = []
    gens = []

    n_threads = 24

    graph_stats_list = compute_graph_stats(graph_list)
    pgds += compute_pgd(graph_stats_list, n_threads)
    trials += [trial for _ in graph_stats_list]
    gens += [x for x in range(len(graph_stats_list))]

    df_full = construct_full_table(pgds, trials, gens, model)
    df_full.to_csv(output_filename, float_format='%.7f', sep='\t', index=False, na_rep='nan')

    return dataset+model

def parallel_computation(input_path, dataset, model):

    path = os.path.join(input_path, dataset, model)
    input_filenames = [f for f in listdir(path) if isfile(join(path, f))]

    number_of_files = len(input_filenames)
    n_threads = 2

    pbar_inner = tqdm(number_of_files)

    def pbar_update(result):
        pbar_inner.update()
        pbar_inner.set_postfix_str(result)


    # for idx in range(number_of_files):
    #     sublevel_parallel_computation(p_arg[0],p_arg[1],p_arg[2], idx)

    asyncResults = []
    with mp.Pool(n_threads) as innerPool:
        ColorPrint.print_green(f"Starting Pool with {n_threads} threads with {len(parallel_args)} tasks.")
        for idx in range(number_of_files):
            r = innerPool.apply_async(sublevel_parallel_computation, [input_path, dataset, model, idx], callback=pbar_update)
            asyncResults.append(r)
        for r in asyncResults:
            try:
                r.wait()
            except:
                continue

    return model, dataset

if __name__ == '__main__':
    input_path = '/afs/crc.nd.edu/user/t/tford5/infinity-mirror/cleaned/'
    datasets = ['eucore', 'clique-ring-500-4', 'flights']
    models = ['CNRG']

    this_list = [['clique-ring-500-4', 'GCN_AE'], ['clique-ring-500-4', 'Linear_AE'], ['clique-ring-500-4', 'Kronecker'], ['eucore', 'Kronecker'], ['flights', 'GCN_AE'], ['flights', 'Linear_AE'], ['flights', 'Kronecker'], ['tree', 'GCN_AE'], ['tree', 'Linear_AE'], ['chess', 'GCN_AE'], ['chess', 'Linear_AE']]
    this_list.reverse()
    # pgds = []
    # trials = []
    # gens = []
    parallel_args = []
    results = []

    for thing in this_list:
        parallel_args.append([input_path, thing[0], thing[1]])

    pbar = tqdm(len(parallel_args))

    def result_update(result):
        pbar.update()
        pbar.set_postfix_str(result[0]+result[1])

    # sequential implementation
    for p_arg in parallel_args:
        results.append(parallel_computation(p_arg[0],p_arg[1],p_arg[2]))

    # asyncResults = []
    # with mp.Pool(12) as outerPool:
    #     ColorPrint.print_green(f"Starting Pool with {12} threads with {len(parallel_args)} tasks.")
    #     for p_arg in parallel_args:
    #         r = outerPool.apply_async(parallel_computation, p_arg, callback=result_update)
    #         asyncResults.append(r)
    #     for r in asyncResults:
    #         try:
    #             r.wait()
    #         except:
    #             continue

