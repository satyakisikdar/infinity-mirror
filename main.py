import os
import networkx as nx

from src.graph_io import GraphReader, GraphWriter
from src.graph_models import ErdosRenyi, ChungLu, BTER
from src.graph_analysis import GraphStats, make_plot


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

    cl = ChungLu(input_graph=g)
    cl.generate(10)
    print(cl)

    bter = BTER(input_graph=g)
    bter.generate(1)
    print(bter)


def test_graph_stats(g: nx.Graph):
    g_stats = GraphStats(graph=g)
    k_hop = g_stats.k_hop_reachability()
    make_plot(y=k_hop, title=f'Hop-Plot for {g.name}', xlabel='Hops', ylabel='Avg. fraction of reachable nodes')


def main():
    make_dirs()
    graph_reader = GraphReader(filename='./input/eucore.g') # , reindex_nodes=True, first_label=1)
    g = graph_reader.graph
    # test_generators(g)
    test_graph_stats(g)


if __name__ == '__main__':
    main()
