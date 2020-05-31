#!/usr/bin/env python
# coding: utf-8

import os
import sys
import multiprocessing as mp
import threading
import time
import pandas as pd
import numpy as np
import scipy.stats as st

from tqdm import tqdm

#sys.path.append('./../dgonza26/infinity-mirror')
sys.path.append('./../..')

from src.graph_comparison import GraphPairCompare
from src.graph_stats import GraphStats
from src.utils import load_pickle, ColorPrint


def compute_graph_stats(root):
    graph_stats = [GraphStats(graph=node.graph, run_id=1) for node in [root] + list(root.descendants)]
    return graph_stats


def absolute(graph_stats):
    for gs in graph_stats[1:]:
        comparator = GraphPairCompare(graph_stats[0], gs)
        dist = comparator.js_distance()
        yield dist


def sequential(graph_stats):
    prev = graph_stats[0]
    for curr in graph_stats[1:]:
        comparator = GraphPairCompare(prev, curr)
        prev = curr
        dist = comparator.js_distance()
        yield dist


def absolute_js(graph_stats):
    abs_js = [x for x in absolute(graph_stats)]
    return abs_js


def sequential_js(graph_stats):
    seq_js = [x for x in sequential(graph_stats)]
    return seq_js


def length_chain(root):
    return len(root.descendants)


def flatten(L):
    return [item for sublist in L for item in sublist]


def mkdir_output(path):
    if not os.path.isdir(path):
        try:
            os.mkdir(path)
        except OSError:
            print('ERROR: could not make directory {path} for some reason')
    return

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
    if abs_js != []:
        abs_mean, abs_ci = compute_stats(abs_js)
        abs_lower = abs_ci[:, 0]
        abs_upper = abs_ci[:, 1]
    else:
        abs_mean = []
        abs_lower = []
        abs_upper = []
    if seq_js != []:
        seq_mean, seq_ci = compute_stats(seq_js)
        seq_lower = seq_ci[:, 0]
        seq_upper = seq_ci[:, 1]
    else:
        seq_mean = []
        seq_lower = []
        seq_upper = []
    gen = [x + 1 for x in range(len(abs_mean))]

    rows = {'model': model, 'gen': gen, 'abs_mean': abs_mean, 'abs-95%': abs_lower, 'abs+95%': abs_upper, 'seq_mean': seq_mean, 'seq-95%': seq_lower, 'seq+95%': seq_upper}

    df = pd.DataFrame(rows)
    return df

def get_filenames(base_path, dataset, models):
    filenames = []
    print(f'loading {dataset} {models[0]}')
    for model in models:
        path = os.path.join(base_path, dataset, model)
        for subdir, dirs, files in os.walk(path):
            for filename in files:
                if 'seq' in filename and 'rob' not in filename:
                    #print(f'loading {filename}')
                    filenames.append(os.path.join(subdir, filename))
                    # yield load_pickle(os.path.join(subdir, filename))
    ColorPrint.print_bold(f"Found {len(filenames)} graph files to be loaded.")
    return filenames


def load_graph(filename):
    # todo: ask about the slice
    root = load_pickle(filename)
    return root


def parallel_thing(root):
    graph_stats = compute_graph_stats(root)
    local_abs_js = absolute_js(graph_stats)
    local_seq_js = sequential_js(graph_stats)

    return [local_abs_js, local_seq_js]


def driver():
    pass


if __name__ == '__main__':
    base_path = '/data/infinity-mirror'
    dataset = 'eucore'
    models_list = ['BTER']
    num = 10

    for model in models_list:
        models = [model]
        output_path = os.path.join(base_path, dataset, models[0], 'jensen-shannon')
        mkdir_output(output_path)

        filenames = get_filenames(base_path, dataset, models)
        graphs_list = []

        results_lock = threading.Lock()

        # pandas dict variables
        abs_js = []
        seq_js = []

        read_pbar = tqdm(len(filenames), desc="Reading Files", position=0, leave=False)
        work_pbar = tqdm(len(filenames), desc="Processing Files", position=1, leave=True)

        active_reads_Lock = threading.Lock()
        active_reads = 0
        pending_work_Lock = threading.Lock()
        pending_work = 0
        active_work_Lock = threading.Lock()
        active_work = 0


        def read_update(result):
            global active_reads
            global pending_work
            global graphs_list
            with active_reads_Lock:
                active_reads -= 1
            with pending_work_Lock:
                pending_work += 1
            graphs_list.append(result)
            read_pbar.update()


        def work_update(result):
            # store results in global lists
            with results_lock:
                global abs_js
                global seq_js
                #global M

                abs_js.append(result[0])
                seq_js.append(result[1])

            # update work status variables
            global active_work
            with active_work_Lock:
                active_work -= 1
            work_pbar.update()


        work_pool = mp.Pool(num)

        with mp.Pool(num) as read_pool:
            while filenames or graphs_list:
                if active_reads + pending_work + active_work <= num:
                    if filenames:
                        filename = filenames.pop(0)  # take the first item
                        active_reads += 1
                        read_pool.apply_async(load_graph, [filename], callback=read_update)
                        # graphs_list.append(read_update(load_graph(filename)))
                    for idx, graph in enumerate(graphs_list):
                        active_work += 1
                        work_pool.apply_async(parallel_thing, [graph], callback=work_update)
                        graphs_list.pop(idx)
                        pending_work -= 1
                else:
                    for idx, graph in enumerate(graphs_list):
                        active_work += 1
                        work_pool.apply_async(parallel_thing, [graph], callback=work_update)
                        graphs_list.pop(idx)
                        pending_work -= 1
                    #ColorPrint.print_blue(f'Sleeping {active_reads}, {pending_work}, {active_work}')
                    time.sleep(10)
        # wait until everything is off of the queue
        while active_work > 0:
            time.sleep(num)

        work_pool.close()

        df = construct_table(abs_js, seq_js, models[0])
        df.to_csv(f'{output_path}/{dataset}_{models[0]}_js.csv', float_format='%.7f', sep='\t', index=False)
