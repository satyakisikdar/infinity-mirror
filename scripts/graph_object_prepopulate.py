#!/usr/bin/env python
# coding: utf-8

import os
import sys;

sys.path.append('../')

from src.graph_comparison import GraphPairCompare
from src.graph_stats import GraphStats

from tqdm import tqdm

from src.utils import load_pickle, ColorPrint

import multiprocessing as mp

import threading

import time
import pandas as pd

import numpy as np
import scipy.stats as st


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
    print('absolute... ', end='', flush=True)
    abs_js = [x for x in absolute(root)]
    print('done')
    return abs_js


def sequential_js(root):
    print('sequential... ', end='', flush=True)
    seq_js = [x for x in sequential(root)]
    print('done')
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


def load_graph(path):
    root = load_pickle(path)


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

read_pbar = tqdm(len(filenames), desc="Reading Files")
work_pbar = tqdm(len(filenames), desc="Processing Files")

graphs_list = []

active_reads_Lock = threading.Lock()
active_reads = 0
pending_work_Lock = threading.Lock()
pending_work = 0
active_work_Lock = threading.Lock()
active_work = 0


def read_update(result):
    global active_reads
    global pending_work
    with active_reads_Lock:
        active_reads -= 1
    with pending_work_Lock:
        pending_work += 1
    graphs_list.append(result)
    read_pbar.update()


def work_update(local_abs_js, local_seq_js, local_gen, local_M):
    # store results in global lists
    with results_lock:
        global abs_js
        global seq_js
        global gen
        global M

        abs_js.append(local_abs_js)
        seq_js.append(local_seq_js)
        gen += local_gen
        M += local_M

    # update work status variables
    global active_work
    with active_work_Lock:
        active_work -= 1
    work_pbar.update()


def parallel_thing(root):
    pass

if __name__ == '__main__':
    base_path = '/data/infinity-mirror'
    dataset = 'eucore'
    models = ['BTER']
    num = 5

    filenames = get_filenames(base_path, dataset, models)
    graphs_list = []

    results_lock = threading.Lock()

    # pandas dict variables
    abs_js = []
    seq_js = []
    gen = []

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
            #global M

            abs_js.append(result[0])
            seq_js.append(result[1])
            gen += result[2]

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
                    # work_update(parallel_thing(graph))
                    work_pool.apply_async(parallel_thing, [graph], callback=work_update)
                    graphs_list.pop(idx)
                    pending_work -= 1
            else:
                for idx, graph in enumerate(graphs_list):
                    active_work += 1
                    # work_update(parallel_thing(graph))
                    work_pool.apply_async(parallel_thing, [graph], callback=work_update)
                    graphs_list.pop(idx)
                    pending_work -= 1
                ColorPrint.print_blue(f'Sleeping {active_reads}, {pending_work}, {active_work}')
                time.sleep(10)
    # wait until everything is off of the queue
    while active_work > 0:
        time.sleep(num)

    work_pool.close()
