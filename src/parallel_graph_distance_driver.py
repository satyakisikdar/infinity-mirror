import os
import sys;

from src.graph_distance import GraphDistance

sys.path.extend('../')
import networkx as nx
from src.graph_stats import GraphStats
from src.parallel import parallel_async
from src.utils import load_pickle, get_imt_input_directory, walker, ColorPrint


def distance_computation(bucket, dataset, model, trial, filename, stats):
    while iteration <
    GraphDistance(dataset=dataset, trial=trial, model=model, metrics=stats)


    return None

if __name__ == '__main__':
    stat = ['pagerank', 'degree_dist']
    buckets, datasets, models, trials, filenames = walker()
    #datasets, models, trials, filenames = ['eucore']*10, ['BTER']*10, [str(x) for x in range(1, 11)], [f'list_20_{x}.pkl.gz' for x in range(1, 11)]

    args = [(bucket, dataset, model, trial, filename, stat) for bucket, dataset, model, trial, filename in zip(buckets, datasets, models, trials, filenames)]

    parallel_async(stats_computation, args)
