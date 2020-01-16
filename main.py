import argparse
import ast
import glob
import importlib
import os
import time
from typing import Any

import networkx as nx
from joblib import Parallel, delayed
from pathlib import Path

from src.graph_io import GraphReader, SyntheticGraph
from src.graph_stats import GraphStats
from src.infinity_mirror import InfinityMirror
from src.utils import timer, ColorPrint as CP
from src.graph_comparison import GraphPairCompare


def parse_args():
    model_names = {'ErdosRenyi', 'ChungLu', 'BTER', 'CNRG', 'HRG', 'Kronecker', 'UniformRandom'}
    selections = {'best', 'worst', 'median'}

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)  # formatter class shows defaults in help

    # using choices we can control the inputs. metavar='' prevents printing the choices in the help preventing clutter
    parser.add_argument('-i', '--input', help='Input graph', metavar='', nargs='+', required=True)

    parser.add_argument('-m', '--model', help='Model to use', metavar='', choices=model_names, nargs=1, required=True)

    parser.add_argument('-n', '--gens', help='#generations', nargs=1, metavar='', type=int, required=True)

    parser.add_argument('-s', '--sel', help='Selection policy', choices=selections, nargs=1, metavar='', required=True)

    parser.add_argument('-o', '--outdir', help='Name of the output directory', nargs=1, default='output', metavar='')

    parser.add_argument('-p', '--pickle', help='Use pickle?', action='store_true')

    parser.add_argument('-g', '--num_graphs', help='#graphs/generation', default=[10], nargs=1, metavar='', type=int)

    parser.add_argument('-c', '--cores', help='#cores to use', default=[1], nargs=1, metavar='', type=int)

    parser.add_argument('-t', '--trials', help='#trials', nargs=1, metavar='', type=int, required=True)
    return parser.parse_args()


def process_args(args) -> Any:
    """
    Validates args
    :param args:
    :return:
    """
    possible_extensions = {'.g', '.gml', '.txt', '.gml', '.mat'}
    graph_names = {fname[: fname.find(ext)].split('/')[-1]
                   for ext in possible_extensions
                   for fname in glob.glob(f'./input/*{ext}')}
    graph_names.update(set(SyntheticGraph.implemented_methods))  # add the synthetic graph generators

    # check input
    if len(args.input) > 1:
        kind = args.input[0]  # kind of synthetic graph
        assert kind in SyntheticGraph.implemented_methods, f'{kind} not implemented in SyntheticGraph class'

        kwd_args = {}
        for param, val in zip(SyntheticGraph.implemented_methods[kind], args.input[1:]):
            kwd_args[param] = ast.literal_eval(val)

        g = SyntheticGraph(kind, **kwd_args).g
    else:
        g = GraphReader(filename=args.input[0]).graph

    model_name = args.model[0]
    module = importlib.import_module(f'src.graph_models')
    model_obj = getattr(module, model_name)

    return args.sel[0], g, model_obj, int(args.gens[0]), args.pickle, int(args.num_graphs[0])


def make_dirs(gname, model):
    """
    Makes input and output directories if they do not exist already
    :return:
    """
    for dirname in ('input', 'output', 'analysis', 'src/scratch', 'output/pickles', f'output/pickles/{gname}',
                    f'output/pickles/{gname}/{model}'):
        if not Path(f'./{dirname}').exists():
            os.makedirs(f'./{dirname}')


def run_infinity_mirror(args, run_id):
    """
    Creates and runs infinity mirror
    :return:
    """
    selection, g, model, num_gens, use_pickle, num_graphs = process_args(args)

    # process args returns the Class and not an object
    empty_g = nx.empty_graph(1)
    empty_g.name = 'empty'  # create an empty graph as a placeholder
    model_obj = model(
        input_graph=empty_g,
        run_id=run_id)  # this is a roundabout way to ensure the name of GraphModel object is correct

    make_dirs(g.name, model=model_obj.model_name)
    inf = InfinityMirror(selection=selection, initial_graph=g, num_generations=num_gens, model_obj=model,
                         num_graphs=num_graphs, run_id=run_id)
    tic = time.perf_counter()
    inf.run(use_pickle=use_pickle)
    toc = time.perf_counter()

    inf.write_timing_stats(round(toc - tic, 3))
    print(run_id, inf)


@timer
def main():
    CP.print_orange('GCD is disabled')

    args = parse_args()
    num_jobs, num_trials = int(args.cores[0]), int(args.trials[0])

    CP.print_green(f'Running infinity mirror on {num_jobs} cores for {num_trials} trials')

    Parallel(n_jobs=num_jobs, backend="loky")(
        delayed(run_infinity_mirror)(run_id=i + 1, args=args)
        for i in range(num_trials)
    )

    return


if __name__ == '__main__':
    main()
