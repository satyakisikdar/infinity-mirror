#!/usr/bin/env python
# coding: utf-8

import os
import sys;
import multiprocessing as mp
import threading
import time
import pandas as pd
import numpy as np
import scipy.stats as st

from tqdm import tqdm

sys.path.append('./../dgonza26/infinity-mirror')
sys.path.append('..')

from src.graph_comparison import GraphPairCompare
from src.graph_stats import GraphStats
from src.utils import load_pickle, ColorPrint


def absolute(root):
    for node in list(root.descendants):
        comparator = GraphPairCompare(GraphStats(graph=root.graph, run_id=1), GraphStats(graph=node.graph, run_id=1))
        dist = comparator.js_distance()
        yield dist


def sequential(root):
    prev = root
    for node in list(root.descendants):
        comparator = GraphPairCompare(GraphStats(graph=prev.graph, run_id=1), GraphStats(graph=node.graph, run_id=1))
        prev = node
        dist = comparator.js_distance()
        yield dist


def absolute_js(root):
    # print('absolute... ', end='', flush=True)
    abs_js = [x for x in absolute(root)]
    # print('done')
    return abs_js


def sequential_js(root):
    # print('sequential... ', end='', flush=True)
    seq_js = [x for x in sequential(root)]
    # print('done')
    return seq_js


def length_chain(root):
    return len(root.descendants)


def flatten(L):
    return [item for sublist in L for item in sublist]


def stats(js):
    mean = np.mean(js, axis=0)
    ci = []
    for row in np.asarray(js).T:
        ci.append(st.t.interval(0.95, len(row) - 1, loc=np.mean(row), scale=st.sem(row)))
    return np.asarray(mean), np.asarray(ci)


def construct_table(abs_js, seq_js, gen, M):
    # abs_js = [absolute_js(root) for root in roots]
    # seq_js = [sequential_js(root) for root in roots]
    abs_mean, abs_ci = stats(abs_js)
    seq_mean, seq_ci = stats(seq_js)
    # gen = [x for x in range(len(abs_mean))]

    rows = {'model': M, 'gen': gen, 'abs_mean': abs_mean, 'abs-95%': abs_ci[:, 0], 'abs+95%': abs_ci[:, 1],
            'seq_mean': seq_mean, 'seq-95%': seq_ci[:, 0], 'seq+95%': seq_ci[:, 1]}

    df = pd.DataFrame(rows)
    return df


def get_filenames(base_path, dataset, models):
    filenames = []
    for model in models:
        path = os.path.join(base_path, dataset, model)
        for subdir, dirs, files in os.walk(path):
            for filename in files:
                if 'seq' not in filename and 'rob' not in filename:
                    # print(f'loading {filename}')
                    filenames.append(os.path.join(subdir, filename))
                    # yield load_pickle(os.path.join(subdir, filename))
    ColorPrint.print_bold(f"Found {len(filenames)} graph files to be loaded.")
    return filenames


def load_graph(filename):
    # todo: ask about the slice
    root = load_pickle(filename)
    return root


def parallel_thing(model, root):
    local_abs_js = absolute_js(root)
    local_seq_js = sequential_js(root)
    local_gen = [x for x in range(len(abs_js))]
    local_M = [model for _ in range(len(abs_js))]

    return [local_abs_js, local_seq_js, local_gen, local_M]


def driver():
    pass


if __name__ == '__main__':
    input_path = '/data/infinity-mirror'
    output_path = '/data/infinity-mirror/'
    dataset = 'tree'
    models = ['BTER']
    model = models[0]

    filenames = get_filenames(input_path, dataset, models)

    graphs_list = []

    results_lock = threading.Lock()

    abs_js = []
    seq_js = []
    gen = []
    M = []

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
            global gen
            global M

            abs_js.append(result[0])
            seq_js.append(result[1])
            gen += result[2]
            M += result[3]

        # update work status variables
        global active_work
        with active_work_Lock:
            active_work -= 1
        work_pbar.update()


    work_pool = mp.Pool(5)

    with mp.Pool(5) as read_pool:
        while filenames or graphs_list:
            if active_reads + pending_work + active_work <= 5:
                if filenames:
                    filename = filenames.pop(0)  # take the first item
                    active_reads += 1
                    read_pool.apply_async(load_graph, [filename], callback=read_update)
                    # graphs_list.append(read_update(load_graph(filename)))
                for idx, graph in enumerate(graphs_list):
                    active_work += 1
                    work_pool.apply_async(parallel_thing, (model, graph), callback=work_update)
                    graphs_list.pop(idx)
                    pending_work -= 1
            else:
                for idx, graph in enumerate(graphs_list):
                    active_work += 1
                    work_pool.apply_async(parallel_thing, (model, graph), callback=work_update)
                    graphs_list.pop(idx)
                    pending_work -= 1
                # ColorPrint.print_blue(f'Sleeping {active_reads}, {pending_work}, {active_work}')
                time.sleep(10)
    # wait until everything is off of the queue
    while graphs_list:
        time.sleep(5)

    work_pool.close()

    df = construct_table(abs_js, seq_js, gen, model)
    df.to_csv(f'data-JS/tf_test_{dataset}_{model}_js.csv', float_format='%.7f', sep='\t', index=False)