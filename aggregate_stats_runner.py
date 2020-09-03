import gzip
import json
import numpy as np

from src.aggregate_stats import compute_stat
from src.utils import walker_michigan

all_datasets = ['chess', 'clique-ring-500-4', 'eucore', 'flights', 'tree']
all_models = ['BTER', 'BUGGE', 'Chung-Lu', 'CNRG', 'Erdos-Renyi', 'GCN_AE', 'GraphRNN', 'HRG', 'Kronecker', 'Linear_AE', 'NetGAN', 'SBM']
all_stats = ['b_matrix', 'degree_dist', 'laplacian_eigenvalues', 'netlsd', 'pagerank', 'pgd_graphlet_counts']
all_post = ['degree_js', 'lambda_dist', 'netlsd', 'pagerank_js', 'pgd_rgfd', 'portrait_js']

def main():
    dataset = 'clique-ring-500-4'
    stat = 'pgd_graphlet_counts'

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
        print(df.head())

    return

if __name__ == '__main__':
    main()
