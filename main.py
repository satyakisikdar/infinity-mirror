import argparse
import ast
import glob
import importlib
import os

import networkx as nx

from src.graph_io import GraphReader, SyntheticGraph
from src.graph_models import *
from src.graph_stats import GraphStats
from src.infinity_mirror import InfinityMirror
from src.utils import timer


# TODO: parallelize stuff - even more?

def parse_args():
    model_names = {'ErdosRenyi', 'ChungLu', 'BTER', 'CNRG', 'HRG', 'Kronecker'}

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)  # formatter class shows defaults in help

    # using choices we can control the inputs. metavar='' prevents printing the choices in the help preventing clutter
    parser.add_argument('-i', '--input', help='Input graph', metavar='', nargs='+', required=True)

    parser.add_argument('-m', '--model', help='Model to use', metavar='', choices=model_names, nargs=1, required=True)

    parser.add_argument('-n', '--num_gens', help='Number of generations', default=5, nargs=1, metavar='', type=int)

    parser.add_argument('-o', '--outdir', help='Name of the output directory', default='output', metavar='')

    parser.add_argument('-p', '--use_pickle', help='Use pickle?', action='store_true')
    return parser.parse_args()


def process_args(args):
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

    return g, model_obj, int(args.num_gens[0]), args.use_pickle


def make_dirs():
    """
    Makes input and output directories if they do not exist already
    :return:
    """
    for dirname in ('input', 'output', 'analysis', 'src/scratch', 'output/pickles'):
        if not os.path.exists(f'./{dirname}'):
            os.makedirs(f'./{dirname}')


@timer
def test_infinity_mirror(g: nx.Graph):
    inf = InfinityMirror(initial_graph=g, num_generations=3, model_obj=ChungLu)  # CNRG seems to create rings
    inf.run(use_pickle=False)
    print('Note: GCD is disabled')
    inf.plot()
    print(inf)


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
    # g_stats['diameter']
    # print('blah')
    # g_stats['diameter']

    # k_hop = g_stats.k_hop_reach()
    # deg_dist = g_stats.degree_dist(normalized=True)
    # cc_by_deg = g_stats.clustering_coefficients_by_degree()
    #
    #
    # make_plot(y=k_hop, title=f'Hop-Plot for {g.name}', xlabel='Hops', ylabel='Avg. fraction of reachable nodes')
    # make_plot(y=deg_dist, title=f'Degree-Dist for {g.name}', xlabel='Degree $k$', ylabel='Count of nodes',
    #           kind='scatter')

    # make_plot(y=cc_by_deg, title=f'Avg Clustering-Coeff by Degree (k)', xlabel='Degree $k$',
    #           ylabel='Avg clustering coefficient', kind='scatter')
    # print(deg_dist)


def main():
    args = parse_args()
    g, model, num_gens, use_pickle = process_args(args)
    print('GCD is disabled')
    inf = InfinityMirror(initial_graph=g, num_generations=num_gens, model_obj=model)
    inf.run(use_pickle=use_pickle)
    print(inf)

    return


if __name__ == '__main__':

    main()
