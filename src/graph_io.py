'''
Grpah i/o helpers
'''
import networkx as nx
from pathlib import Path
import numpy as np
import pyintergraph as pig
import igraph as ig

from src.utils import ColorPrint as CP
from src.utils import check_file_exists


class GraphReader:
    """
    Class for graph reader
    .g /.txt: graph edgelist
    .gml, .gexf for Gephi
    .mat for adjacency matrix
    """
    def __init__(self, filename: str, gname: str=''):
        """
        :param filename: path to input file
        :param gname: name of the graph
        """
        self.possible_extensions = ['.g', '.gexf', '.gml', '.txt', '.mat']
        self.filename = filename
        self.path = Path(filename)
        assert check_file_exists(self.path), f'Path: {self.path} does not exist'

        if gname != '':
            self.gname = gname
        else:
            self.gname = self.path.stem
        self.graph = self._read()
        assert self.graph.name != '', 'Graph name is empty'

    def _read(self) -> nx.Graph:
        """
        Reads the graph based on its extension
        :return:
        """
        extension = self.path.suffix

        assert extension in self.possible_extensions, f'Invalid extension {extension}, supported extensions: ' \
                                                      f'{self.possible_extensions}'

        graph = nx.Graph()
        str_path = str(self.path)

        if extension in ('.g', '.txt'):
            graph: nx.Graph = nx.read_edgelist(str_path)

        elif extension == '.gml':
            graph: nx.Graph = nx.read_gml(str_path)

        elif extension == '.gexf':
            graph: nx.Graph = nx.read_gexf(str_path)

        elif extension == '.mat':
            mat = np.loadtxt(fname=str_path, dtype=bool)
            graph: nx.Graph = nx.from_numpy_array(mat)

        graph.name = self.gname
        CP.print_green(f'Reading {self.gname} from {self.path} with n={graph.order():,}, m={graph.size():,}')
        return graph

    def __str__(self):
        return f'<GraphReader object> graph: {self.gname}, path: {str(self.path)} n={self.graph.order():,}, m={self.graph.size()}'

    def __repr__(self):
        return str(self)


class GraphWriter:
    """
    Class for writing graphs, expects a networkx graph as input
    """
    def __init__(self, graph: nx.Graph, path: str, fmt: str='', gname: str=''):
        self.graph = graph

        if self.graph == '':
            self.graph.name = gname

        assert self.graph.name != '', 'Graph name is empty'

        self.path = Path(path)
        if fmt == '':  # figure out extension from filename
            self.fmt = self.path.suffix
        else:
            self.fmt = fmt
        self._write()

    def _write(self):
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

    def __str__(self):
        return f'<GraphWriter object> graph: {self.graph}, path: {str(self.path)} n={self.graph.order():,}, m={self.graph.size():,}'

    def __repr__(self):
        return str(self)


def networkx_to_graphtool(nx_G: nx.Graph):
    return pig.nx2gt(nx_G, labelname='node_label')


def graphtool_to_networkx(gt_G):
    graph = pig.InterGraph.from_graph_tool(gt_G)
    return graph.to_networkX()


def networkx_to_igraph(nx_G: nx.Graph):
    graph = pig.InterGraph.from_networkX(nx_G)
    return graph.to_igraph()


def igraph_to_networkx(ig_G: ig.Graph):
    graph = pig.InterGraph.from_igraph(ig_G)
    return graph.to_networkX()

