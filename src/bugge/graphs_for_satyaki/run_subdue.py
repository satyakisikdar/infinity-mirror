from time import time 
import os
import subprocess
import networkx as nx
from sys import argv 

def read_graph(fname):
    g = nx.read_edgelist(fname, nodetype=int, create_using=nx.Graph())
    print(fname)
    name = fname.split('/')[-1]
    name = '_'.join(name.split('_')[: -2])
    g.name = name
    print(f'Read {name}, n = {g.order()}, m = {g.size()}')
    return g


def subdue(g):
    print('Starting SUBDUE....')

    name = g.name
    g = nx.convert_node_labels_to_integers(g, first_label=1)
    g.name = name

    with open('../subdue/{}_sub.g'.format(g.name), 'w') as f:
        for u in sorted(g.nodes()):
            f.write('\nv {} v'.format(u))

        for u, v in g.edges():
            f.write('\nd {} {} e'.format(u, v))

    start_time = time()

    completed_process = subprocess.run('cd ../subdue; ./subdue {}_sub.g'.format(g.name),
                                       shell=True, stdout=subprocess.PIPE)

    print('SUBDUE ran in {} secs'.format(round(time() - start_time, 3)))

def main():
    if len(argv) < 2:
        print('Provide path to edge list')
        return 
    g = read_graph(argv[1])

    # subdue(g)


if __name__ == '__main__':
    main()