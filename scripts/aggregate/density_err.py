from collections import Counter
import os
import sys; sys.path.append('./../../')
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

def load_data(base_path, dataset, model, seq_flag, rob_flag):
    path = os.path.join(base_path, dataset, model)
    for subdir, dirs, files in os.walk(path):
        for filename in files:
            if '.csv' not in filename and 'jensen-shannon' not in subdir:
                #if ((seq_flag and 'seq' in filename) and (not seq_flag and 'seq' not in filename)) and ((rob_flag and 'rob' in filename) and (not rob_flag and 'rob' not in filename)):
                if 'seq' not in filename and 'rob' not in filename:
                    print(f'\tloading {subdir} {filename} ... ', end='', flush=True)
                    pkl = load_pickle(os.path.join(subdir, filename))
                    print('done')
                    yield pkl, model

def mkdir_output(path):
    if not os.path.isdir(path):
        try:
            os.mkdir(path)
        except OSError:
            print('ERROR: could not make directory {path} for some reason')
    return

def unravel(root):
    if type(root) is list:
        return root
    else:
        graphs = [node.graph for node in [root] + list(root.descendants)]
        return graphs

def absolute(graphs):
    density0 = nx.density(graphs[0])
    for G in graphs[1:]:
        density = nx.density(G) - density0
        yield density

def sequential(graphs):
    prev = graphs[0]
    for curr in graphs[1:]:
        density = nx.density(curr) - nx.density(prev)
        prev = curr
        yield density

def absolute_density(graphs):
    print('\t\tabsolute... ', end='', flush=True)
    abs_densities = [x for x in absolute(graphs)]
    print('done')
    return abs_densities

def sequential_density(graphs):
    print('\t\tsequential... ', end='', flush=True)
    seq_densities = [x for x in sequential(graphs)]
    print('done')
    return seq_densities

def length_chain(root):
    return len(root.descendants)

def flatten(L):
    return [item for sublist in L for item in sublist]

def compute_stats(densities):
    #padding = max(len(l) for l in js)
    #for idx, l in enumerate(js):
    #    while len(js[idx]) < padding:
    #        js[idx] += [np.NaN]
    for idx, l in enumerate(densities):
        if l == []:
            densities[idx] = [0]
    print(densities)
    mean = np.nanmean(densities, axis=0)
    return np.asarray(mean)

def construct_table(abs_densities, model):
    abs_mean = compute_stats(abs_densities)
    gen = [x + 1 for x in range(len(abs_mean))]

    rows = {'model': model, 'gen': gen, 'abs_mean': abs_mean}

    df = pd.DataFrame(rows)
    return df

def main():
    base_path = '/data/infinity-mirror/cleaned-new-new'
    #input_path = '/home/dgonza26/infinity-mirror/input'
    dataset = 'clique-ring-500-4'
    models = ['Kronecker']

    for model in models:
        #output_path = os.path.join(base_path, dataset, models[0], 'jensen-shannon')
        output_path = '/data/infinity-mirror/stats/density'
        mkdir_output(output_path)

        abs_densities = []
        gen = []
        for root, model in load_data(base_path, dataset, model, True, False):
            graphs = unravel(root)
            assert graphs != []
            abs_densities.append(absolute_density(graphs))

        df = construct_table(abs_densities, model)
        df.to_csv(f'{output_path}/{dataset}_{model}_density.csv', float_format='%.7f', sep='\t', index=False, na_rep='nan')
        print(f'wrote: {output_path}/{dataset}_{model}_density.csv')

    return

main()
