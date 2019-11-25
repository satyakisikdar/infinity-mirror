import os

from src.graph_io import GraphReader, GraphWriter
from src.graph_models import ErdosRenyi, ChungLu, BTER

def make_dirs():
    '''
    Makes input and output directories if they do not exist already
    :return:
    '''
    for dirname in ('input', 'output', 'analysis'):
        if not os.path.exists(f'./{dirname}'):
            os.makedirs(f'./{dirname}')


def test_generators(g):
    er = ErdosRenyi(input_graph=g)
    er.generate(10)
    print(er)

    cl = ChungLu(input_graph=g)
    cl.generate(10)
    print(cl)

    bter = BTER(input_graph=g)
    bter.generate(1)
    print(bter)


def main():
    make_dirs()
    graph_reader = GraphReader(filename='./input/karate.mat')
    g = graph_reader.graph
    test_generators(g)

if __name__ == '__main__':
    main()