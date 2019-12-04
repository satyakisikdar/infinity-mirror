import os
import networkx as nx

from src.graph_io import GraphReader, GraphWriter
from src.graph_models import *
from src.graph_stats import GraphStats
from src.utils import make_plot


def make_dirs():
    """
    Makes input and output directories if they do not exist already
    :return:
    """
    for dirname in ('input', 'output', 'analysis'):
        if not os.path.exists(f'./{dirname}'):
            os.makedirs(f'./{dirname}')


def test_generators(g: nx.Graph):
    er = ErdosRenyi(input_graph=g)
    er.generate(10)
    print(er)

    # hrg = HRG(input_graph=g)
    # hrg.generate(10)
    # print(hrg)

    # cnrg = CNRG(input_graph=g)
    # cnrg.generate(10)
    # print(cnrg)

    # cl = ChungLu(input_graph=g)
    # cl.generate(10)
    # print(cl)
    #
    # bter = BTER(input_graph=g)
    # bter.generate(1)
    # print(bter)


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
    make_dirs()
    graph_reader = GraphReader(filename='./input/karate.g', reindex_nodes=True, first_label=0)
    g = graph_reader.graph
    # test_generators(g)
    test_graph_stats(g)


if __name__ == '__main__':
    # try:
    #     main()
    # except Exception as e:
    #     print(e)
    main()
