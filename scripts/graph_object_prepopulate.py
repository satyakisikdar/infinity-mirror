#!/usr/bin/env python
# coding: utf-8

import os
import sys; sys.path.append('./../dgonza26/infinity-mirror')

from tqdm import tqdm

from src.utils import load_pickle, ColorPrint

import multiprocessing as mp

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
    root = load_pickle(filename)
    return root


input_path = '/data/infinity-mirror'
output_path = '/data/infinity-mirror/'
dataset = 'flights'
models = ['Erdos-Renyi']
model = models[0]

filenames = get_filenames(input_path, dataset, models)

pbar = tqdm(len(filenames))

graphs_list = []

def update_progress(result):
    # graphs_list.append(result)
    pbar.update()


with mp.Pool(5) as pool:
    for filename in filenames:
        graphs_list.append(pool.apply_async(load_graph, [filename], callback=update_progress))
    for obj in graphs_list:
        obj.wait()

# single processed version for debugging
# for filename in tqdm(filenames):
#     graphs_list.append(load_graph(filename))
    #

