import gzip
import json
import numpy as np
import pandas as pd
import scipy.stats as stats

from src.aggregate_stats import compute_stat
from src.utils import walker_michigan

all_datasets = ['chess', 'clique-ring-500-4', 'eucore', 'flights', 'tree']
all_models = ['BTER', 'BUGGE', 'Chung-Lu', 'CNRG', 'Erdos-Renyi', 'GCN_AE', 'GraphRNN', 'HRG', 'Kronecker', 'Linear_AE', 'NetGAN', 'SBM']
all_stats = ['b_matrix', 'degree_dist', 'laplacian_eigenvalues', 'netlsd', 'pagerank', 'pgd_graphlet_counts']
all_post = {'b_matrix': 'portrait_js', 'degree_dist': 'degree_js', 'laplacian_eigenvalues': 'lambda_dist', 'netlsd': 'netlsd', 'pagerank': 'pagerank_js', 'pgd_graphlet_counts': 'pgd_rgfd'}

def confidence_interval(x):
    lower, upper = stats.t.interval(0.95, len(x) - 1, loc=np.mean(x), scale=stats.sem(x))
    return lower, upper

def main():
    dataset = 'clique-ring-500-4'
    stat = 'degree_dist'
    post = all_post[stat]

    dataframes = []
    for model in all_models:
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

    df_concat = pd.concat(dataframes)

    df_mean = df_concat.groupby(['dataset', 'model', 'gen'])[post].mean().reset_index()
    df_ci_lower = df_concat.groupby(['dataset', 'model', 'gen'])[post].apply(lambda x: confidence_interval(x)[0]).reset_index()
    df_ci_upper = df_concat.groupby(['dataset', 'model', 'gen'])[post].apply(lambda x: confidence_interval(x)[1]).reset_index()

    df_mean['ci_lower'] = df_ci_lower[post]
    df_mean['ci_upper'] = df_ci_upper[post]

    print(df_mean)

    return

if __name__ == '__main__':
    main()
