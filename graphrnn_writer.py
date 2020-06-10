from collections import Counter
import os
import sys#; sys.path.append('./../../')
import pickle
import numpy as np
import pandas as pd
import networkx as nx
import scipy.stats as st
import multiprocessing as mp
from pathlib import Path
from src.Tree import TreeNode
from src.utils import load_pickle
from src.graph_stats import GraphStats
from src.graph_comparison import GraphPairCompare
from src.utils import verify_dir

def init(filename: str, gname: str = '', reindex_nodes: bool = False, first_label: int = 0, take_lcc: bool = True) -> nx.Graph:
    """
    :param filename: path to input file
    :param gname: name of the graph
    """
    possible_extensions = ['.g', '.gexf', '.gml', '.txt', '.mat']
    filename = filename
    path = Path(filename)
    #assert check_file_exists(path), f'Path: "{self.path}" does not exist'

    if gname == '':
        gname = path.stem

    graph: nx.Graph = read(path, gname)
    graph: nx.Graph = preprocess(graph, reindex_nodes=reindex_nodes, first_label=first_label, take_lcc=take_lcc)
    assert graph.name != '', 'Graph name is empty'
    return graph

def read(path: str, gname: str) -> nx.Graph:
    """
    Reads the graph based on its extension
    returns the largest connected component
    :return:
    """
    #CP.print_blue(f'Reading "{self.gname}" from "{self.path}"')
    extension = path.suffix
    #assert extension in possible_extensions, f'Invalid extension "{extension}", supported extensions: {possible_extensions}'

    str_path = str(path)

    if extension in ('.g', '.txt'):
        graph: nx.Graph = nx.read_edgelist(str_path, nodetype=int)

    elif extension == '.gml':
        graph: nx.Graph = nx.read_gml(str_path)

    elif extension == '.gexf':
        graph: nx.Graph = nx.read_gexf(str_path)

    elif extension == '.mat':
        mat = np.loadtxt(fname=str_path, dtype=bool)
        graph: nx.Graph = nx.from_numpy_array(mat)
    else:
        raise (NotImplementedError, f'{extension} not supported')

    graph.name = gname
    return graph

def preprocess(graph: nx.Graph, reindex_nodes: bool, first_label: int = 0, take_lcc: bool = True) -> nx.Graph:
    """
    Preprocess the graph - taking the largest connected components, re-index nodes if needed
    :return:
    """
    #CP.print_none('Pre-processing graph....')
    #CP.print_none(f'Original graph "{self.gname}" n:{self.graph.order():,} '
    #              f'm:{self.graph.size():,} #components: {nx.number_connected_components(self.graph)}')

    if take_lcc and nx.number_connected_components(graph) > 1:
        ## Take the LCC
        component_sizes = [len(c) for c in sorted(nx.connected_components(graph), key=len, reverse=True)]

        #CP.print_none(f'Taking the largest component out of {len(component_sizes)} components: {component_sizes}')

        graph_lcc = nx.Graph(graph.subgraph(max(nx.connected_components(graph), key=len)))

        perc_nodes = graph_lcc.order() / graph.order() * 100
        perc_edges = graph_lcc.size() / graph.size() * 100
        #CP.print_orange(f'LCC has {print_float(perc_nodes)}% of nodes and {print_float(perc_edges)}% edges in the original graph')

        graph = graph_lcc

    selfloop_edges = list(nx.selfloop_edges(graph))
    if len(selfloop_edges) > 0:
        #CP.print_none(f'Removing {len(selfloop_edges)} self-loops')
        graph.remove_edges_from(selfloop_edges)  # remove self-loops

    if reindex_nodes:
        # re-index nodes, stores the old label in old_label
        graph = nx.convert_node_labels_to_integers(graph, first_label=first_label,
                                                        label_attribute='old_label')
        #CP.print_none(
        #    f'Re-indexing nodes to start from {first_label}, old labels are stored in node attr "old_label"')

    #CP.print_none(f'Pre-processed graph "{self.gname}" n:{self.graph.order():,} m:{self.graph.size():,}')
    return graph

def load_data(base_path, dataset):
    path = os.path.join(base_path, 'GraphRNN', f'{dataset}_size10_ratio5')
    for subdir, dirs, files in os.walk(path):
        for filename in files:
            if '1000' in filename:
                print(f'loading {subdir} {filename} ... ', end='', flush=True)
                pkl = load_pickle(os.path.join(subdir, filename))
                print('done')
                yield pkl, int(filename.split('_')[1])
    return

def flatten(L):
    return [item for sublist in L for item in sublist]

def construct_full_table(abs_lambda, seq_lambda, model, trials):
    gen = []
    for t in trials:
        gen += [x + 1 for x in range(len(t))]

    rows = {'model': model, 'trial': flatten(trials), 'gen': gen, 'abs': abs_lambda}#, 'seq': seq_lambda}

    df = pd.DataFrame(rows)
    return df

def main():
    base_path = '/data/infinity-mirror/cleaned-new'
    input_path = '/home/dgonza26/infinity-mirror/input'
    dataset = 'tree'
    model = 'GraphRNN'

    output_path = os.path.join(base_path, dataset, model)
    verify_dir(output_path)

    R = [(root, generation) for root, generation in load_data(os.path.join(base_path, 'cleaned'), dataset)]
    R.sort(key=lambda x: x[1])
    R = [root for (root, generation) in R]

    if dataset == 'clique-ring-500-4':
        G = nx.ring_of_cliques(500, 4)
    else:
        G = init(os.path.join(input_path, f'{dataset}.g'))

    # add the initial graph and transpose the list
    roots = [[G] + list(r) for r in zip(*R)]

    for idx, r in enumerate(roots, 1):
        print('writing', os.path.join(output_path, f'list_{len(r)}_{idx}.pkl.gz'))
        pkl.dump(r, os.path.join(output_path, f'list_{len(r)}_{idx}.pkl.gz'))

    return

main()
