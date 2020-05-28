import csv
import pickle
from collections import namedtuple
from typing import Any, List, Dict, Union

import networkx as nx
from tqdm import tqdm

from src.Tree import TreeNode, LightTreeNode
from src.graph_comparison import GraphPairCompare
from src.graph_models import *
from src.graph_stats import GraphStats
from src.utils import borda_sort, ColorPrint as CP, load_pickle, check_file_exists

Stats = namedtuple('Stats',
                   'name id graph score')  # stores the different stats for each graph. name: name of metric, id: graph_id
GraphStatDouble = namedtuple('GraphStatDouble',
                             'graph stats')  # graph: NetworkX object, stat: dictionary of comparison stats with the input
GraphStatTriple = namedtuple('GraphStatTriple',
                             'best worst median')  # stores the best, worst, and median graphs and their stats (GraphStat double)


class InfinityMirror:
    """
    Class for InfinityMirror
    """
    __slots__ = ('initial_graph', 'num_generations', 'num_graphs', 'model', 'initial_graph_stats', 'graphs',
                 '_metrics', 'graphs_pickle_path', 'run_id', 'rewire')

    def __init__(self, initial_graph: nx.Graph, model_obj: Any, num_generations: int,
                 num_graphs: int, run_id: int, r: float) -> None:
        self.run_id = run_id
        self.initial_graph: nx.Graph = initial_graph  # the initial starting point H_0
        self.num_graphs: int = num_graphs  # number of graphs per generation
        self.num_generations: int = num_generations  # number of generations

        self.model = model_obj # (input_graph=self.initial_graph, run_id=self.run_id)
        self.model.input_graph = self.initial_graph
        self.model.run_id = run_id # initialize and fit the model

        self.initial_graph_stats: GraphStats = GraphStats(run_id=run_id, graph=self.initial_graph)
        self._metrics: List[str] = ['deltacon0', 'lambda_dist', 'pagerank_cvm', 'node_diff', 'edge_diff', 'pgd_pearson',
                                    'pgd_spearman', 'degree_cvm']  # list of metrics  ## GCD is removed
        self.rewire = int(r * 100)
        self.graphs_pickle_path: str = f'./output/pickles/{self.initial_graph.name}/{self.model.model_name}/list_{self.num_generations}_{self.run_id}'
        self.graphs: List[nx.Graph] = []  # stores the list of graphs - one per generation
        if r != 0:
            self.graphs_pickle_path += f'_{r}'
        return

    def __str__(self) -> str:
        return f'({self.run_id}) model: "{self.model.model_name}"  initial graph: "{self.initial_graph.name}"  #gens: {self.num_generations}'

    def __repr__(self) -> str:
        return str(self)


    def run(self, use_pickle: bool) -> None:
        """
        New runner - uses LightTreeNode objects, so no graph comparison
        :param use_pickle:
        :return:
        """
        pickle_ext = '.pkl.gz'
        if use_pickle and check_file_exists(self.graphs_pickle_path + pickle_ext):
            CP.print_green(f'Using pickle at "{self.graphs_pickle_path + pickle_ext}"')
            graphs = load_pickle(self.graphs_pickle_path + pickle_ext)
            assert isinstance(list, LightTreeNode), 'Invalid TreeNode format, needs to be a LightTreeNode object'
            self.graphs = graphs
            return

        tqdm.write(
            f'Running Infinity Mirror on "{self.initial_graph.name}" {self.initial_graph.order(), self.initial_graph.size()} "{self.model.model_name}" {self.num_generations} generations')
        pbar = tqdm(total=self.num_generations, bar_format='{l_bar}{bar}|[{elapsed}<{remaining}]', ncols=50)

        self.initial_graph.level = 0
        self.graphs = [self.initial_graph]

        for i in range(self.num_generations):
            if i == 0:
                curr_graph = self.initial_graph  # current graph is the initial graph

            level = i + 1
            try:
                self.model.update(new_input_graph=curr_graph)  # update the model
            except Exception as e:
                print(f'Model fit failed {e}')
                self.graphs_pickle_path += f'_failed-{level}'  # append the failed to filenam
                break

            try:
                generated_graphs = self.model.generate(num_graphs=self.num_graphs,
                                                       gen_id=level)  # generate a new set of graphs
            except Exception as e:
                print(f'Generation failed {e}')
                self.graphs_pickle_path += f'_failed-{level}'  # append the failed to filename
                break

            curr_graph = generated_graphs[0]  # we are only generating one graph
            curr_graph.name = f'{self.initial_graph.name}_{level}_{self.run_id}'
            curr_graph.gen = level
            self.graphs.append(curr_graph)

            pbar.update(1)

        pbar.close()
        CP.print_green(f'List of Graphs is pickled at "{self.graphs_pickle_path + pickle_ext}"')
        pickle.dump(self.graphs, open(self.graphs_pickle_path + pickle_ext, 'wb'), protocol=-1)  # use highest possible protocol
        return

    def write_timing_stats(self, time_taken) -> None:
        """
        Write timing stats into a csv
        Write model info and timing info
        :return:
        """
        fieldnames = ['run_id', 'gname', 'model', 'sel', 'gens', 'time']

        stats_file = './output/timing_stats.csv'
        if not check_file_exists(stats_file):  # initialize the file with headers
            writer = csv.DictWriter(open(stats_file, 'w'), fieldnames=fieldnames)
            writer.writeheader()

        with open(stats_file, 'a') as fp:
            writer = csv.DictWriter(fp, fieldnames=fieldnames)
            writer.writerow({'run_id': self.run_id, 'gname': self.initial_graph.name, 'model': self.model.model_name,
                             'gens': self.num_generations, 'time': time_taken})

        return

    def write_fail_stats(self, level) -> None:
        """
        Write fail stats into a csv
        :return:
        """
        fieldnames = ['run_id', 'gname', 'model', 'sel', 'gens', 'level']

        fail_file = './output/fail_stats.csv'
        if not check_file_exists(fail_file):  # initialize the file with headers
            writer = csv.DictWriter(open(fail_file, 'w'), fieldnames=fieldnames)
            writer.writeheader()

        with open(fail_file, 'a') as fp:
            writer = csv.DictWriter(fp, fieldnames=fieldnames)
            writer.writerow({'run_id': self.run_id, 'gname': self.initial_graph.name, 'model': self.model.model_name,
                            'gens': self.num_generations, 'level': level})

        return
