import argparse
import ast
import glob
import importlib
import os
import time
from typing import Any

import networkx as nx
from joblib import Parallel, delayed

from src.graph_io import GraphReader, SyntheticGraph
from src.graph_models import *
from src.graph_stats import GraphStats
from src.infinity_mirror import InfinityMirror
from src.utils import timer, ColorPrint as CP

# TODO: parallelize stuff - even more?
# TODO: write a stats dump file - write the config and the times in a csv

def parse_args():
    model_names = {'ErdosRenyi', 'ChungLu', 'BTER', 'CNRG', 'HRG', 'Kronecker', 'UniformRandom'}

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)  # formatter class shows defaults in help

    # using choices we can control the inputs. metavar='' prevents printing the choices in the help preventing clutter
    parser.add_argument('-i', '--input', help='Input graph', metavar='', nargs='+', required=True)

    parser.add_argument('-m', '--model', help='Model to use', metavar='', choices=model_names, nargs=1, required=True)

    parser.add_argument('-n', '--num_gens', help='Number of generations', default=[5], nargs=1, metavar='', type=int)

    parser.add_argument('-s', '--selection', help='Selection policy', choices=('best', 'worst', 'median'), nargs=1,
                        metavar='', required=True)

    parser.add_argument('-o', '--outdir', help='Name of the output directory', nargs=1, default='output', metavar='')

    parser.add_argument('-p', '--use_pickle', help='Use pickle?', action='store_true')

    parser.add_argument('-g', '--num_graphs', help='Number of graphs per generation', default=[10], nargs=1, metavar='',
                        type=int)
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

    return args.selection[0], g, model_obj, int(args.num_gens[0]), args.use_pickle, int(args.num_graphs[0])


def make_dirs(gname, model):
    """
    Makes input and output directories if they do not exist already
    :return:
    """
    for dirname in ('input', 'output', 'analysis', 'src/scratch', 'output/pickles', f'output/pickles/{gname}',
                    f'output/pickles/{gname}/{model}'):
        if not os.path.exists(f'./{dirname}'):
            os.makedirs(f'./{dirname}')


def test_generators(g: nx.Graph):
    er = ErdosRenyi(input_graph=g)
    er.generate(10, gen_id=0)
    print(er)

    kron = Kronecker(input_graph=g)
    kron.generate(2, gen_id=0)
    print(kron)

    hrg = HRG(input_graph=g)
    hrg.generate(10, gen_id=0)
    print(hrg)

    cnrg = CNRG(input_graph=g)
    cnrg.generate(10, gen_id=0)
    print(cnrg)

    cl = ChungLu(input_graph=g)
    cl.generate(10, gen_id=0)
    print(cl)

    bter = BTER(input_graph=g)
    bter.generate(1, gen_id=0)
    print(bter)


def test_graph_stats(g: nx.Graph):
    g_stats = GraphStats(graph=g)
    g_stats._calculate_all_stats()
    print(g_stats)


def run_infinity_mirror(run_id):
    """
    Creates and runs infinity mirror
    :return:
    """
    args = parse_args()
    selection, g, model, num_gens, use_pickle, num_graphs = process_args(args)

    # process args returns the Class and not an object
    empty_g = nx.empty_graph(1)
    empty_g.name = 'empty'  # create an empty graph as a placeholder
    model_obj = model(
        input_graph=empty_g, run_id=run_id)  # this is a roundabout way to ensure the name of GraphModel object is correct

    make_dirs(g.name, model=model_obj.model_name)
    print('GCD is disabled')

    inf = InfinityMirror(selection=selection, initial_graph=g, num_generations=num_gens, model_obj=model,
                         num_graphs=num_graphs, run_id=run_id)
    tic = time.perf_counter()
    inf.run(use_pickle=False)
    toc = time.perf_counter()

    inf.write_timing_stats(round(toc - tic, 3))
    print(run_id, inf)

@timer
def main():
    CP.print_orange('GCD is disabled')

    Parallel(n_jobs=10, backend="multiprocessing")(
        delayed(run_infinity_mirror)(run_id=i+1)
        for i in range(10)
    )

    return


if __name__ == '__main__':
    main()
