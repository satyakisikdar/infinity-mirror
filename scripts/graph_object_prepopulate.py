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


def work_update():
    # # store results in global lists
    # with results_lock:
    #     global abs_js
    #     global seq_js
    #     global gen
    #     global M
    #
    #     abs_js.append(local_abs_js)
    #     seq_js.append(local_seq_js)
    #     gen += local_gen
    #     M += local_M

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
