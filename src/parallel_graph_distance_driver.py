import os
import sys;
import pandas as pd

from src.graph_distance import GraphDistance

sys.path.extend('../')
import networkx as nx
from src.graph_stats import GraphStats
from src.parallel import parallel_async
from src.utils import load_pickle, get_imt_input_directory, walker, ColorPrint, ensure_dir, walker_texas_ranger


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
            GD.compute_distances([stat])
            results = GD.stats[stat]

            row = {}
            row.update({'dataset': dataset, 'model': model, 'trial': trial, 'iteration': iteration, stat: results})
            rows.append(row)


        results_df = pd.DataFrame(rows)
        # results_df.to_csv(path_or_buf=f'{output_dir}/{stat}_{trial}.csv', index=False)

    return results_df

if __name__ == '__main__':

    datasets = ['eucore']
    models = ['BTER']
    stats = ['pagerank_js']

    # buckets, datasets, models, trials, filenames = walker()
    # datasets, models, trials = ['eucore']*4, ['BTER']*4, [1,2,3,4]

    for stat in stats:
        for dataset in datasets:
            for model in models:
                trials = walker_texas_ranger(dataset, model, stat='pagerank', unique=True)
                args = [[dataset, model, trial, stat] for trial in trials]
                results = parallel_async(distance_computation, args, num_workers=16)
                df = pd.concat(results)

                output_dir = f'/data/infinity-mirror/output/distances/{dataset}/{model}/{stat}/'
                ensure_dir(output_dir, recursive=False)
                df.to_csv(output_dir+f'{dataset}_{model}_{stat}.csv')
                # for arg in args:
                #     distance_computation(*arg)
