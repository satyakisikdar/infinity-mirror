import os

from src.graph_io import GraphReader, GraphWriter

def make_dirs():
    '''
    Makes input and output directories if they do not exist already
    :return:
    '''
    for dirname in ('input', 'output'):
        if not os.path.exists(f'./{dirname}'):
            os.makedirs(f'./{dirname}')

def main():
    make_dirs()
    graph_reader = GraphReader(filename='./input/karate.mat')
    graph = graph_reader.graph

    GraphWriter(graph=graph, path='./output/karate.gml')

if __name__ == '__main__':
    main()