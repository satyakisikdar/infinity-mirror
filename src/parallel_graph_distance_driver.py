import os
import sys;
import pandas as pd

from src.graph_distance import GraphDistance

sys.path.extend('../')
import networkx as nx
from src.graph_stats import GraphStats
from src.parallel import parallel_async
from src.utils import load_pickle, get_imt_input_directory, walker, ColorPrint, ensure_dir


def distance_computation(dataset, model, trial, stats):
    if not isinstance(stats, list):
        stats = [stats]

    for stat in stats:
        columns = ['dataset', 'model', 'trial', 'iteration'].append(stat)

        GD = GraphDistance(dataset=dataset, trial=trial, model=model, metrics=[stat], iteration=None)
        GD.set_root_object(GD.implemented_metrics[stat])
        total_iterations = GD.total_iterations

        rows = []

        for iteration in range(total_iterations):
            GD.set_iteration(iteration=iteration)
            GD.compute_distances([stat])
            results = GD.stats[stat]
            if not isinstance(results, list):
                results = [results]

            row = {}
            row.update({'dataset': dataset, 'model': model, 'trial': trial, 'iteration': iteration, stat: results})
            rows.append(row)
        output_dir = f'/data/infinity-mirror/output/distances/{dataset}/{model}/{stat}/'
        ensure_dir(output_dir, recursive=False)

        results_df = pd.DataFrame(rows)
        results_df.to_csv(path_or_buf=f'{output_dir}/{stat}_{trial}.csv')

    return None

if __name__ == '__main__':
    stat = ['pagerank_js']
    # buckets, datasets, models, trials, filenames = walker()
    datasets, models, trials = ['eucore'], ['BTER'], [8]

    args = [(dataset, model, trial, stat) for dataset, model, trial in zip(datasets, models, trials)]

    # parallel_async(distance_computation, args, num_workers=1)
    for arg in args:
        distance_computation(*arg)