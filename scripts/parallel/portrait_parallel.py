import os
import sys;
from itertools import combinations

from tqdm import tqdm

#from pgd import mkdir_output

sys.path.append('./../../')
from os import listdir
from os.path import isfile, join
import multiprocessing as mp

# from src.pgd import mkdir_output

import pandas as pd
from src.utils import load_pickle
from src.portrait.portrait_divergence import portrait_divergence


def load_data(input_path, dataset, model):
    path = os.path.join(input_path, dataset, model)
    input_filenames = [f for f in listdir(path) if isfile(join(path, f))]
    # print(input_filenames)
    for filename in input_filenames:
        pkl = load_pickle(os.path.join(path, filename))
        trial = filename.split('_')[2].strip('.pkl.gz')
        yield pkl, trial

def parallel_computation(input_path, output_path, dataset, model):
    #mkdir_output(os.path.join(output_path,model))
    output_filename = f'{os.path.join(output_path,model)}/{dataset}_{model}_portrait.csv'
    # if the data already exists, just don't redo it.
    if os.path.isfile(output_filename):
        df = pd.read_csv(output_filename, dtype={'model': str, 'gen': int, 'trial_id': int, 'portrait': float}, sep='\t')
        df['name'] = [dataset]*df.shape[0]

        df.to_csv(f'{os.path.join(output_path,model)}/portrait_{dataset}_{model}.csv', float_format='%.7f', sep='\t', index=False, na_rep='nan')
        return df, model, dataset

    rows = {col: [] for col in cols}
    for root, trial in tqdm(load_data(input_path, dataset, model), desc=model+dataset):
        if isinstance(root, list):
            root_graph = root[0]
            descendants = root[1:]
        else:
            root_graph = root.graph
            descendants = [tnode.graph for tnode in root.descendants]

        for i, desc_graph in enumerate(descendants, 1):
            d = portrait_divergence(G=root_graph, H=desc_graph)

            rows['model'].append(model)
            rows['name'].append(dataset)
            rows['gen'].append(i)
            rows['trial_id'].append(int(trial))
            rows['portrait'].append(d)

    df = pd.DataFrame(rows)
    # save out incremental work in case something goes wrong
    df.to_csv(f'{os.path.join(output_path,model)}/portrait_{dataset}_{model}.csv', float_format='%.7f', sep='\t', index=False, na_rep='nan')
    return df, model, dataset

if __name__ == '__main__':
    input_path = '/data/infinity-mirror/cleaned-new/'
    output_path = '/data/infinity-mirror/stats/portrait/'
    datasets = ['tree']
    models = ['GraphRNN']

    cols = ['name', 'model', 'trial_id', 'gen', 'portrait']
    parallel_args = []
    results = []
    # make the list of parameters that will be passed to the pool workers
    for dataset in datasets:
        for model in models:
            parallel_args.append([input_path, output_path, dataset, model])

    pbar = tqdm(len(parallel_args))

    # sequential implementation
    # for p_arg in parallel_args:
    #     results.append(parallel_computation(p_arg[0],p_arg[1],p_arg[2], p_arg[3]))

    def result_update(result):
        pbar.update()
        pbar.set_postfix_str(result[1]+result[2])
        results.append(result[0])

    asyncResults = []
    with mp.Pool(24) as outerPool:
        for p_arg in parallel_args:
            r = outerPool.apply_async(parallel_computation, p_arg, callback=result_update)
            asyncResults.append(r)
        for r in asyncResults:
            try:
                r.wait()
            except:
                continue

    df = pd.concat(results)
    #mkdir_output(output_path)
    df.to_csv(f'{output_path}/merged_portrait.csv', float_format='%.7f', sep='\t', index=False, na_rep='nan')
