import os
import sys;
import pandas as pd

from src.graph_distance import GraphDistance

sys.path.extend('../')
import networkx as nx
from src.graph_stats import GraphStats
from src.parallel import parallel_async
from src.utils import load_pickle, get_imt_input_directory, walker, ColorPrint


def distance_computation(dataset, model, trial, stats):

    for stat in stats:

        results_df = pd.DataFrame(columns=['dataset', 'model', 'trial', 'iteration']+stats)
        GD = GraphDistance(dataset=dataset, trial=trial, model=model, metrics=stats)
        total_iterations = GD.total_iterations

        for iteration in range(total_iterations):
            GD.set_iteration(iteration=iteration)
            results = GD.compute_distances()
            row = pd.DataFrame(columns=results_df.columns, data=[[dataset, model, trial, iteration].extend(results)])
            results_df.append(row)
        results_df.st.to_csv(path_or_buf=f'/data/infinity_mirror/distances/{dataset}/{model}/{stat}/{stat}_{trial}.csv')

    return None

if __name__ == '__main__':
    stat = ['pagerank']
    # buckets, datasets, models, trials, filenames = walker()
    datasets, models, trials = ['eucore']*10, ['BTER']*10, [str(x) for x in range(1, 11)]

    args = [(dataset, model, trial, stat) for dataset, model, trial in zip(datasets, models, trials)]

    parallel_async(distance_computation, args, num_workers=1)
