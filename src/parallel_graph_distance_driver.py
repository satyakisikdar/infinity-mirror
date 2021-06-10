import os
import sys;
from pathlib import Path

sys.path.extend('../')
import pandas as pd

from src.graph_distance import GraphDistance

import networkx as nx
from src.graph_stats import GraphStats
from src.parallel import parallel_async
from src.utils import load_pickle, get_imt_input_directory, walker, ColorPrint, ensure_dir, walker_texas_ranger, \
    get_imt_output_directory


def distance_computation(dataset, model, trial, stats):
    if not isinstance(stats, list):
        stats = [stats]

    for stat in stats:
        GD = GraphDistance(dataset=dataset, trial=trial, model=model, metrics=[stat], iteration=None)
        GD.set_root_object(GD.implemented_metrics[stat])
        total_iterations = GD.total_iterations

        rows = []

        for iteration in range(total_iterations+1):
            GD.set_iteration(iteration=iteration)
            try:
                GD.compute_distances([stat])
            except Exception as e:
                print(f'Error in distance computation {e}')
                continue
            results = GD.stats[stat]

            row = {}
            row.update({'dataset': dataset, 'model': model, 'trial': trial, 'iteration': iteration, stat: results})
            rows.append(row)

        results_df = pd.DataFrame(rows)
        # results_df.to_csv(path_or_buf=f'{output_dir}/{stat}_{trial}.csv', index=False)

    return results_df

# TODO: fix support for HRG (no objects to contatenate)
if __name__ == '__main__':
    implemented_metrics = {'pagerank_js': 'pagerank', 'degree_js': 'degree_dist', 'pgd_distance': 'pgd_graphlet_counts',
                            'netlsd_distance': 'netlsd', 'portrait_divergence': 'b_matrix',
                           'lambda_distance': 'laplacian_eigenvalues'}

    # datasets = ['clique-ring-500-4', 'eucore', 'flights', 'tree']

    # models = ['Chung-Lu', 'CNRG', 'SBM', 'Erdos-Renyi', 'BUGGE', 'HRG'][: 1]
    models = ['BTER', 'BUGGE', 'Chung-Lu', 'CNRG', 'Erdos-Renyi', 'Kronecker', 'HRG', 'SBM', 'GCN_AE', 'Linear_AE',
              'NetGAN', 'GraphRNN']
    #stats = ['pagerank_js', 'degree_js', 'pgd_distance', 'netlsd_distance', 'lambda_distance', 'portrait_divergence']
    stats = ['degree_js', 'pagerank_js', 'lambda_distance', 'portrait_divergence']
    # datasets, models, trials, filenames = walker()
    datasets = ['eucore', 'chess', 'flights', 'cond-mat', 'enron']#[-2: -1]

    datasets = ['cond-mat']
    # stats = ['pagerank_js']
    # models = ['CNRG']

    for dataset in datasets:
        for model in models:
            for stat in stats:
                ColorPrint.print_green(f'computing {stat} distances for {dataset} {model}')
                trials = walker_texas_ranger(dataset, model, stat=implemented_metrics[stat], unique=True)
                args = [[dataset, model, trial, stat] for trial in trials]
                # exit(-1)
                try:
                    results = parallel_async(distance_computation, args, num_workers=8)
                    df = pd.concat(results)
                except Exception as e:
                    ColorPrint.print_red(f'Error, for {dataset!r} {model!r} {stat!r} {e}')
                    continue

                # output_dir = f'/data/infinity-mirror/output/distances/{dataset}/{model}/{stat}/'
                output_dir = Path(get_imt_output_directory()) / 'new-distances' / stat
                ensure_dir(output_dir, recursive=True)
                df.to_csv(output_dir / f'{dataset}_{model}_{stat}.csv')
                # for arg in args:
                #     distance_computation(*arg)
