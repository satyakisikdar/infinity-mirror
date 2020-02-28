import networkx as nx 
from sys import argv 
from glob import glob

def convert_edgelist(fname):
    g = nx.DiGraph()
    name = fname.split('/')[-1]
    name = '_'.join(name.split('_')[: -2])
    g.name = name
    with open(fname) as fp:
        for line in fp:
            if line.startswith('#'):
                continue
            stuff = list(map(int, line.split()))
            if len(stuff) > 1:
                u = stuff[0]
                for v in stuff[1: ]:
                     g.add_edge(u, v)

    print(f'Edgelist written at ./edge_lists/{name}.g')
    nx.write_edgelist(g, f'./edge_lists/{name}.g', data=False)

for fname in glob('./*.edge_list'):
    convert_edgelist(fname)