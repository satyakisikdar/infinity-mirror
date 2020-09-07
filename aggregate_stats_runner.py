import gzip
import json
import numpy as np
import pandas as pd
import scipy.stats as stats

import warnings
warnings.filterwarnings("ignore")

from src.aggregate_stats import compute_mono_stat, compute_bi_stat
from src.utils import walker_michigan

all_datasets = ['chess', 'clique-ring-500-4', 'eucore', 'flights', 'tree']
all_models = ['BTER', 'BUGGE', 'Chung-Lu', 'CNRG', 'Erdos-Renyi', 'GCN_AE', 'GraphRNN', 'HRG', 'Kronecker', 'Linear_AE', 'NetGAN', 'SBM']
all_stats = ['b_matrix', 'degree_dist', 'laplacian_eigenvalues', 'netlsd', 'pagerank', 'pgd_graphlet_counts', 'average_path_length', 'average_clustering']
all_post = {'b_matrix': 'portrait_js', 'degree_dist': 'degree_js', 'laplacian_eigenvalues': 'lambda_dist', 'netlsd': 'netlsd', 'pagerank': 'pagerank_js', 'pgd_graphlet_counts': 'pgd_rgfd', 'average_path_length': 'avg_pl', 'average_clustering': 'avg_clustering', 'apl_cc': ['clu', 'pl']}

def confidence_interval(x):
    lower, upper = stats.t.interval(0.95, len(x) - 1, loc=np.mean(x), scale=stats.sem(x))
    return lower, upper

def main():
    dataset = 'chess'
    stat = 'apl_cc'
    post = all_post[stat]
    dataframes = []

    print(f'aggregating {stat} for {dataset}')

    for model in all_models:
        print(f'\tstarting {model}... ', end='', flush=True)

        if stat == 'apl_cc':
            clu = {}
            pl = {}

            for filepath, trial, gen in walker_michigan(dataset, model, 'average_clustering'):
                with gzip.open(filepath) as f:
                    data = json.loads(f.read())
                try:
                    clu[trial][gen] = data
                except KeyError as e:
                    clu[trial] = {}
                    clu[trial][gen] = data

            for filepath, trial, gen in walker_michigan(dataset, model, 'average_path_length'):
                with gzip.open(filepath) as f:
                    data = json.loads(f.read())
                try:
                    pl[trial][gen] = data
                except KeyError as e:
                    pl[trial] = {}
                    pl[trial][gen] = data

            df = compute_bi_stat(dataset, model, stat, clu, pl)
            dataframes.append(df)
            print(f'done')
        else:
            agg = {}

            for filepath, trial, gen in walker_michigan(dataset, model, stat):
                with gzip.open(filepath) as f:
                    data = json.loads(f.read())
                try:
                    agg[trial][gen] = data
                except KeyError as e:
                    agg[trial] = {}
                    agg[trial][gen] = data

            df = compute_mono_stat(dataset, model, stat, agg)
            dataframes.append(df)
            print(f'done')

    df_concat = pd.concat(dataframes)

    df_full = df_concat.groupby(['dataset', 'model', 'gen'])[post].mean().reset_index()
    #print(df_full.head())

    if stat != 'apl_cc':
        df_ci_lower = df_concat.groupby(['dataset', 'model', 'gen'])[post].apply(lambda x: confidence_interval(x)[0]).reset_index()
        df_ci_upper = df_concat.groupby(['dataset', 'model', 'gen'])[post].apply(lambda x: confidence_interval(x)[1]).reset_index()

        df_full['ci_lower'] = df_ci_lower[post]
        df_full['ci_upper'] = df_ci_upper[post]

    if stat == 'apl_cc':
        post = 'apl_cc'
    df_full = df_full.astype({'gen': int})
    df_full.to_csv(f'dataframes/{dataset}_{post}.csv', sep='\t', index=False, na_rep='nan')
    print(f'wrote {dataset}_{post}.csv to dataframes/')

    return

if __name__ == '__main__':
    main()
