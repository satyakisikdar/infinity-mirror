import os
import sys; sys.path.extend('../')
import networkx as nx
from src.graph_stats import GraphStats
from src.parallel import parallel_async
from src.utils import load_pickle, get_imt_input_directory, walker, ColorPrint

def stats_computation(bucket, dataset, model, trial, filename, stats):
    path = os.path.join(get_imt_input_directory(), bucket, dataset, model, filename)
    graph_list = load_pickle(path)
    assert isinstance(graph_list, list), f'Expected type "list" and got type {type(graph_list)}.'
    assert all(isinstance(g, nx.Graph) for g in graph_list), f'Expected a list of nx.Graph and got disappointed instead.'

    ColorPrint.print_orange(f'{filename} has length {len(graph_list)}')

    for idx, G in enumerate(graph_list):
        gs_obj = GraphStats(graph=G, dataset=dataset, model=model, trial=trial, iteration=idx)
        gs_obj.write_stats_jsons(stats=stats)

    return None

if __name__ == '__main__':
    stat = ['pagerank', 'degree_dist']
    buckets, datasets, models, trials, filenames = walker()
    #datasets, models, trials, filenames = ['eucore']*10, ['BTER']*10, [str(x) for x in range(1, 11)], [f'list_20_{x}.pkl.gz' for x in range(1, 11)]

    args = [(bucket, dataset, model, trial, filename, stat) for bucket, dataset, model, trial, filename in zip(buckets, datasets, models, trials, filenames)]

    parallel_async(stats_computation, args)
