from collections import Counter
import os
import sys;

from tqdm import tqdm

sys.path.append('./../../')
import pickle
import numpy as np
import pandas as pd
import networkx as nx
import scipy.stats as st
import multiprocessing as mp
from pathlib import Path
from src.Tree import TreeNode
from src.utils import load_pickle, verify_dir, verify_file
from src.graph_stats import GraphStats
from src.graph_comparison import GraphPairCompare


def load_data(, dataset, model):
    path = os.path.join(base_path, dataset, model)
    for subdir, dirs, files in os.walk(path):
        for filename in files: #Todo: this thing doesn't actually return the trial id...
            if '.csv' not in filename and 'jensen-shannon' not in subdir:
                #if ((seq_flag and 'seq' in filename) and (not seq_flag and 'seq' not in filename)) and ((rob_flag and 'rob' in filename) and (not rob_flag and 'rob' not in filename)):
                if 'seq' not in filename and 'rob' not in filename:
                    print(f'\tloading {subdir} {filename} ... ', end='', flush=True)
                    pkl = load_pickle(os.path.join(subdir, filename))
                    print('done')
                    yield pkl, model

def unravel(root):
    if type(root) is list:
        return root
    else:
        graphs = [node.graph for node in [root] + list(root.descendants)]
        return graphs


def parallel_computation(input_path, dataset, model):
    local_output = output_path + f"{dataset}/{model}"
    verify_dir(local_output)
    rows = {col: [] for col in cols}
    filename = f'{output_path}/graph_desc_{dataset}_{model}.csv'
    for graph_list, trial in tqdm(load_data(input_path, dataset, model)):
        graphs = unravel(graph_list)
        original_graph = graphs[0]

        for idx, graph in enumerate(graphs):
            rows["name"].append(dataset)
            rows['model'] .append(model)
            rows["trial"].append(trial)
            rows['gen'].append(idx)
            rows['nodes'].append(graph.order())
            rows['edges'].append(graph.size())
            rows["density"].append(nx.density(graph))
            rows["density_signed_difference"].append(nx.density(graph) - nx.density(original_graph))
            rows["edges_signed_difference"].append(graph.size() - original_graph.size())
            rows["nodes_signed_difference"].append(graph.order() - original_graph.order())

    df = pd.DataFrame(rows)    # save out incremental work in case something goes wrong
    df.to_csv(filename, float_format='%.7f', sep='\t', index=False, na_rep='nan')
    return df



if __name__ == '__main__':
    input_path = '/afs/crc.nd.edu/user/t/tford5/infinity-mirror/cleaned/'
    output_path = '/afs/crc.nd.edu/user/t/tford5/infinity-mirror/output/density/'
    verify_dir(output_path)
    datasets = ['clique-ring-500-4']
    models = ['Kronecker']

    cols = ["name", "model", "trial", "gen", "nodes", "edges", "density", "density_signed_difference", "nodes_signed_difference", "edges_signed_difference"]

    parallel_args = []
    results = []
    # make the list of parameters that will be passed to the pool workers
    for dataset in datasets:
        for model in models:
            parallel_args.append([input_path, dataset, model])

    pbar = tqdm(len(parallel_args))

    # sequential implementation
    for p_arg in parallel_args:
        results.append(parallel_computation(p_arg[0],p_arg[1], p_arg[2]))

    def result_update(result):
        pbar.update()
        pbar.set_postfix_str(result[1]+result[2])
        results.append(result[0])

    # asyncResults = []
    # with mp.Pool(24) as outerPool:
    #     for p_arg in parallel_args:
    #         r = outerPool.apply_async(parallel_computation, p_arg, callback=result_update)
    #         asyncResults.append(r)
    #     for r in asyncResults:
    #         try:
    #             r.wait()
    #         except:
    #             continue

    df = pd.concat(results)
    df.to_csv(f'{output_path}/{datasets}_{model}_density.csv', float_format='%.7f', sep='\t', index=False, na_rep='nan')
    print(f'wrote: {output_path}/{dataset}_{model}_density.csv')