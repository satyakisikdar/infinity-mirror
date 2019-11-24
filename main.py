import os

from src.graph_io import GraphReader, GraphWriter
from src.graph_models import ErdosRenyi, ChungLu

def make_dirs():
    '''
    Makes input and output directories if they do not exist already
    :return:
    '''
    for dirname in ('input', 'output'):
        if not os.path.exists(f'./{dirname}'):
            os.makedirs(f'./{dirname}')


def test_generators(g):
    er = ErdosRenyi(input_graph=g)
    er.fit()
    er.generate_graphs(10)
    print(er)

    cl = ChungLu(input_graph=g)
    cl.fit()
    cl.generate_graphs(10)
    print(cl)


def main():
    make_dirs()
    graph_reader = GraphReader(filename='./input/karate.mat')
    g = graph_reader.graph
    test_generators(g)

if __name__ == '__main__':
    main()