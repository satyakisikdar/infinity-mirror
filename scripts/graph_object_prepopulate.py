#!/usr/bin/env python
# coding: utf-8

import os
import sys; sys.path.append('./../dgonza26/infinity-mirror')

from tqdm import tqdm

from src.utils import load_pickle, ColorPrint

import multiprocessing as mp

import threading

import time

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
    return root

def do_thing_function(idx, graph):
    ColorPrint.print_green(f'Did a thing.')
    return idx

input_path = '/data/infinity-mirror'
output_path = '/data/infinity-mirror/'
dataset = 'flights'
models = ['Erdos-Renyi']
model = models[0]

filenames = get_filenames(input_path, dataset, models)

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
    global active_work
    with active_work_Lock:
        active_work -= 1
    work_pbar.update()

work_pool = mp.Pool(5)

with mp.Pool(5) as read_pool:
    while filenames:
        filename = filenames.pop(0) # take the first item
        active_reads += 1
        read_pool.apply_async(load_graph, [filename], callback=read_update)
        for idx, graph in enumerate(graphs_list):
            active_work += 1
            work_pool.apply_async(do_thing_function, (idx,graph), callback=work_update())
            graphs_list.pop(idx)
            pending_work -= 1
        if active_reads+pending_work+active_work > 5:
            time.sleep(10)

work_pool.close()

# single processed version for debugging
# for filename in tqdm(filenames):
#     graphs_list.append(load_graph(filename))
    #

