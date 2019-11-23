from src.graph_io import GraphReader, GraphWriter

def main():
    graph_reader = GraphReader(filename='./input/karate.mat')
    graph = graph_reader.graph

    GraphWriter(graph=graph, path='./output/karate.gml')

if __name__ == '__main__':
    main()