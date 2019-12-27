"""
Grpah i/o helpers
"""
from pathlib import Path

import networkx as nx
import numpy as np

from src.utils import ColorPrint as CP, check_file_exists, print_float
from src.Graph import CustomGraph


class GraphReader:
    """
    Class for graph reader
    .g /.txt: graph edgelist
    .gml, .gexf for Gephi
    .mat for adjacency matrix
    """
    def __init__(self, filename: str, gname: str='', reindex_nodes: bool=False, first_label: int=0, take_lcc: bool=True) -> None:
        """
        :param filename: path to input file
        :param gname: name of the graph
        """
        self.possible_extensions = ['.g', '.gexf', '.gml', '.txt', '.mat']
        self.filename = filename
        self.path = Path(filename)
        assert check_file_exists(self.path), f'Path: "{self.path}" does not exist'

        if gname != '':
            self.gname = gname
        else:
            self.gname = self.path.stem

        self.graph: CustomGraph = self._read()
        self._preprocess(reindex_nodes=reindex_nodes, first_label=first_label, take_lcc=take_lcc)
        assert self.graph.name != '', 'Graph name is empty'
        return

    def _read(self) -> CustomGraph:
        """
        Reads the graph based on its extension
        returns the largest connected component
        :return:
        """
        CP.print_blue(f'Reading "{self.gname}" from "{self.path}"')
        extension = self.path.suffix
        assert extension in self.possible_extensions, f'Invalid extension "{extension}", supported extensions: ' \
                                                      f'{self.possible_extensions}'

        str_path = str(self.path)

        if extension in ('.g', '.txt'):
            graph: CustomGraph = nx.read_edgelist(str_path)

        elif extension == '.gml':
            graph: CustomGraph = nx.read_gml(str_path)

        elif extension == '.gexf':
            graph: CustomGraph = nx.read_gexf(str_path)

        elif extension == '.mat':
            mat = np.loadtxt(fname=str_path, dtype=bool)
            graph: CustomGraph = nx.from_numpy_array(mat)
        else:
            raise(NotImplementedError, f'{extension} not supported')

        graph.name = self.gname
        return CustomGraph(graph)

    def _preprocess(self, reindex_nodes: bool, first_label: int=0, take_lcc: bool=True) -> None:
        """
        Preprocess the graph - taking the largest connected components, re-index nodes if needed
        :return:
        """
        CP.print_green('Pre-processing graph....')
        CP.print_blue(f'Original graph "{self.gname}" n:{self.graph.order():,} '
                        f'm:{self.graph.size():,} #components: {nx.number_connected_components(self.graph)}')

        if take_lcc and nx.number_connected_components(self.graph) > 1:
            ## Take the LCC
            component_sizes = [len(c) for c in sorted(nx.connected_components(self.graph), key=len, reverse=True)]

            CP.print_green(f'Taking the largest component out of {len(component_sizes)} components: {component_sizes}')

            graph_lcc = self.graph.subgraph(max(nx.connected_components(self.graph), key=len))

            perc_nodes = graph_lcc.order() / self.graph.order() * 100
            perc_edges = graph_lcc.size() / self.graph.size() * 100
            CP.print_orange(f'LCC has {print_float(perc_nodes)}% of nodes and {print_float(perc_edges)}% edges in the original graph')

            self.graph = CustomGraph(graph_lcc)

        if reindex_nodes:
            # re-index nodes, stores the old label in old_label
            self.graph = CustomGraph(nx.convert_node_labels_to_integers(self.graph, first_label=first_label,
                                                            label_attribute='old_label'))
            CP.print_green(f'Re-indexing nodes to start from {first_label}, old labels are stored in node attr "old_label"')

        CP.print_green(f'Removing multi-edges and self-loops')
        self.graph = CustomGraph(self.graph)  # make it into a simple graph
        self.graph.remove_edges_from(nx.selfloop_edges(self.graph))  # remove self-loops

        CP.print_blue(f'Pre-processed graph "{self.gname}" n:{self.graph.order():,} m:{self.graph.size():,}')
        return

    def __str__(self) -> str:
        return f'<GraphReader object> graph: {self.gname}, path: {str(self.path)} n={self.graph.order():,}, m={self.graph.size()}'

    def __repr__(self) -> str:
        return str(self)


class SyntheticGraph:
    """
    Container for Synthetic graphs
    """
    implemented_methods = {'chain': {'n'}, 'tree': {'r', 'h'}, 'ladder': {'n'}, 'circular_ladder': {'n'}, 'ring': {'n'},
                           'ring_of_cliques': {'n', 'k'}, 'grid': {'m', 'n'}, 'erdos_renyi': {'n', 'p', 'seed'}}
    def __init__(self, kind, **kwargs):
        self.kind = kind
        assert kind in SyntheticGraph.implemented_methods, f'Generator {kind} not implemented. Implemented methods: {self.implemented_methods.keys()}'
        self.args = kwargs
        self.g = self._make_graph()

    def _make_graph(self) -> CustomGraph:
        """
        Makes the graph
        :return:
        """
        assert self.implemented_methods[self.kind].issubset(self.args), f'Improper args {self.args.keys()}, need: {self.implemented_methods[self.kind]}'

        if self.kind == 'chain':
            g = nx.path_graph(self.args['n'])
            name = f'chain_{g.order()}'
        elif self.kind == 'tree':
            g = nx.balanced_tree(self.args['r'], self.args['h'])
            name = f"tree_{self.args['r']}_{self.args['h']}"
        elif self.kind == 'ladder':
            g = nx.ladder_graph(self.args['n'])
            name = f'ladder_graph_{g.order()}'
        elif self.kind == 'circular_ladder':
            g = nx.circular_ladder_graph(self.args['n'])
            name = f'circular_ladder_graph_{g.order()}'
        elif self.kind == 'ring_of_cliques':
            g = nx.ring_of_cliques(self.args['n'], self.args['k'])
            name = f"clique_ring_{self.args['n']}_{self.args['k']}"
        elif self.kind == 'grid':
            g = nx.grid_2d_graph(self.args['m'], self.args['n'])
            name = f"grid_{self.args['m']}_{self.args['n']}"
        elif self.kind == 'erdos_renyi':
            g = nx.erdos_renyi_graph(n=self.args['n'], p=self.args['p'], seed=self.args['seed'])
            name = f"erdos_renyi_{self.args['n']}_{self.args['p']}"
        else:
            name = ''
            raise NotImplementedError(f'Improper kind: {self.kind}')
        g = CustomGraph(g)
        g.name = name
        return g


class GraphWriter:
    """
    Class for writing graphs, expects a networkx graph as input
    """
    def __init__(self, graph: CustomGraph, path: str, fmt: str='', gname: str=''):
        self.graph: CustomGraph = graph

        if self.graph == '':
            self.graph.name = gname

        assert self.graph.name != '', 'Graph name is empty'

        self.path = Path(path)
        if fmt == '':  # figure out extension from filename
            self.fmt = self.path.suffix
        else:
            self.fmt = fmt
        self._write()

    def _write(self) -> None:
        """
        write the graph into the format
        :return:
        """
        extension = self.path.suffix
        str_path = str(self.path)

        if extension in ('.g', '.txt'):
            nx.write_edgelist(path=str_path, G=self.graph)

        elif extension == '.gml':
            nx.write_gml(path=str_path, G=self.graph)

        elif extension == '.gexf':
            nx.write_gexf(path=str_path, G=self.graph)

        elif extension == '.mat':
            mat = nx.to_numpy_matrix(self.graph, dtype=int)
            np.savetxt(fname=self.path, X=mat, fmt='%d')

        CP.print_blue(f'Wrote {self.graph.name} to {self.path} with n={self.graph.order():,}, m={self.graph.size():,}')
        return

    def __str__(self) -> str:
        return f'<GraphWriter object> graph: {self.graph}, path: {str(self.path)} n={self.graph.order():,}, m={self.graph.size():,}'

    def __repr__(self) -> str:
        return str(self)


try:
    import pyintergraph as pig
except ImportError as e:
    print(e)

try:
    import igraph as ig
except ImportError as e:
    print(e)

def networkx_to_graphtool(nx_G: CustomGraph):
    return pig.nx2gt(nx_G, labelname='node_label')


def graphtool_to_networkx(gt_G):
    graph = pig.InterGraph.from_graph_tool(gt_G)
    return graph.to_networkX()


def networkx_to_igraph(nx_G: CustomGraph):
    graph = pig.InterGraph.from_networkX(nx_G)
    return graph.to_igraph()


def igraph_to_networkx(ig_G: ig.Graph):
    graph = pig.InterGraph.from_igraph(ig_G)
    return graph.to_networkX()

