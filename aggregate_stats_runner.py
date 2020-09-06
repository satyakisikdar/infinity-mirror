import gzip
import json
import numpy as np
import pandas as pd
import scipy.stats as stats

import warnings
warnings.filterwarnings("ignore")

from src.aggregate_stats import compute_stat
from src.utils import walker_michigan, latex_printer

all_datasets = ['chess', 'clique-ring-500-4', 'eucore', 'flights', 'tree']
all_models = ['BTER', 'BUGGE', 'Chung-Lu', 'CNRG', 'Erdos-Renyi', 'GCN_AE', 'GraphRNN', 'HRG', 'Kronecker', 'Linear_AE', 'NetGAN', 'SBM']
all_stats = ['b_matrix', 'degree_dist', 'laplacian_eigenvalues', 'netlsd', 'pagerank', 'pgd_graphlet_counts', 'average_path_length', 'average_clustering']
all_post = {'b_matrix': 'portrait_js', 'degree_dist': 'degree_js', 'laplacian_eigenvalues': 'lambda_dist', 'netlsd': 'netlsd', 'pagerank': 'pagerank_js', 'pgd_graphlet_counts': 'pgd_rgfd', 'average_path_length': 'avg_pl', 'average_clustering': 'avg_clustering'}

def confidence_interval(x):
    lower, upper = stats.t.interval(0.95, len(x) - 1, loc=np.mean(x), scale=stats.sem(x))
    return lower, upper

def main():
    dataset = 'clique-ring-500-4'
    #stat = 'average_path_length'
    stat = 'average_clustering'
    post = all_post[stat]

    print(f'aggregating {stat} for {dataset}')

    dataframes = []
    for model in all_models:
        print(f'\tstarting {model}... ', end='', flush=True)
        agg = {}

        for filepath, trial, gen in walker_michigan(dataset, model, stat):
            with gzip.open(filepath) as f:
                data = json.loads(f.read())
            try:
                agg[trial][gen] = data
            except KeyError as e:
                agg[trial] = {}
                agg[trial][gen] = data

        df = compute_stat(dataset, model, stat, agg)
        dataframes.append(df)
        print(f'done')

    df_concat = pd.concat(dataframes)
    #print(df_concat.head())

    df_full = df_concat.groupby(['dataset', 'model', 'gen'])[post].mean().reset_index()
    #print(df_full.head())

    df_ci_lower = df_concat.groupby(['dataset', 'model', 'gen'])[post].apply(lambda x: confidence_interval(x)[0]).reset_index()
    df_ci_upper = df_concat.groupby(['dataset', 'model', 'gen'])[post].apply(lambda x: confidence_interval(x)[1]).reset_index()

    df_full['ci_lower'] = df_ci_lower[post]
    df_full['ci_upper'] = df_ci_upper[post]

    df_full = df_full.astype({'gen': int})
    df_full.to_csv(f'dataframes/{dataset}_{post}.csv', sep='\t', index=False, na_rep='nan')
    print(f'wrote {dataset}_{post}.csv to dataframes/')

    return

if __name__ == '__main__':
    main()
