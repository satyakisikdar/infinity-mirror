import csv
import pickle
from collections import namedtuple
from typing import Any, List, Dict, Union

import networkx as nx
from tqdm import tqdm

from src.Tree import TreeNode
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
    For each generation, store 3 graphs - best, worst, and the 50^th percentile -
        use ranked choice voting for deciding the winner from 10 graphs <- borda list
        store the three graphs into a tree
    """
    __slots__ = ('initial_graph', 'num_generations', 'num_graphs', 'model', 'initial_graph_stats', 'root',
                 '_metrics', 'root_pickle_path', 'selection', 'run_id', 'rewire')

    def __init__(self, selection: str, initial_graph: nx.Graph, model_obj: Any, num_generations: int,
                 num_graphs: int, run_id: int, r: float) -> None:
        self.run_id = run_id
        self.selection = selection  # kind of selection stategy
        self.initial_graph: nx.Graph = initial_graph  # the initial starting point H_0
        self.num_graphs: int = num_graphs  # number of graphs per generation
        self.num_generations: int = num_generations  # number of generations
        self.model: BaseGraphModel = model_obj(input_graph=self.initial_graph,
                                               run_id=self.run_id)  # initialize and fit the model
        self.initial_graph_stats: GraphStats = GraphStats(run_id=run_id, graph=self.initial_graph)
        self.root: TreeNode = TreeNode('root', graph=self.initial_graph,
                                       stats={})  # root of the tree with the initial graph and empty stats dictionary
        self._metrics: List[str] = ['deltacon0', 'lambda_dist', 'pagerank_cvm', 'node_diff', 'edge_diff', 'pgd_pearson',
                                    'pgd_spearman', 'degree_cvm']  # list of metrics  ## GCD is removed
        self.rewire = int(r * 100)
        self.root_pickle_path: str = f'./output/pickles/{self.initial_graph.name}/{self.model.model_name}/{self.selection}_{self.num_generations}_{self.run_id}'
        if r != 0:
            self.root_pickle_path += f'_{r}'
        return

    def __str__(self) -> str:
        return f'({self.selection}) model: "{self.model.model_name}"  initial graph: "{self.initial_graph.name}"  #gens: {self.num_generations}'

    def __repr__(self) -> str:
        return str(self)

    def _make_graph_stat_double(self, graph, scores, idx) -> GraphStatDouble:
        """
        Makes GraphStatDouble objects
        :param scores:
        :param idx:
        :return:
        """
        stats = {}
        for metric in self._metrics:
            if len(scores[metric]) == 0:  # this is for useless metrics
                stats[metric] = float('inf')
            else:
                stats[metric] = scores[metric][idx].score

        return GraphStatDouble(graph=graph, stats=stats)

    def _get_representative_graph_stat(self, generated_graphs: List[nx.Graph]) -> Union[GraphStatDouble, None]:
        """
        returns the representative graph and its stats
        :param generated_graphs: list of generated graphs
        :return: GraphStatDouble object
        """
        assert len(generated_graphs) != 0, f'generated graphs empty'
        scores: Dict[str, List[Stats]] = {metric: [] for metric in self._metrics}

        graph_comps_list = [GraphPairCompare(gstats1=self.initial_graph_stats, gstats2=GraphStats(gen_graph, run_id=self.run_id))
                            for i, gen_graph in enumerate(generated_graphs)]

        assert isinstance(graph_comps_list, list), 'Graph comp pairs is not a list'
        assert isinstance(graph_comps_list[0], GraphPairCompare), 'Improper object in Graph comp list'

        for i, graph_comp in enumerate(graph_comps_list):
            graph_comp: GraphPairCompare
            for metric in self._metrics:
                score = graph_comp[metric]
                if score == float('inf'):
                    continue
                scores[metric].append(Stats(id=i + 1, graph=graph_comp.graph2, score=graph_comp[metric], name=metric))

        sorted_scores = {key: sorted(val, key=lambda item: item.score) for key, val in scores.items()}

        rankings: Dict[str, List[int]] = {}  # stores the id of the graphs sorted by score

        for metric, stats in sorted_scores.items():
            rankings[metric] = list(map(lambda item: item.id, stats))

        # if all the scores across all the metrics are the same, best, median, and worst are the same graph
        if sum(len(lst) for lst in rankings.values()) == 0:  # empty ranking
            return None

        overall_ranking = borda_sort(rankings.values())  # compute the overall ranking

        if self.selection == 'best':
            idx = overall_ranking[0] - 1  # all indexing is 1 based
        elif self.selection == 'worst':
            idx = overall_ranking[-1] - 1
        elif self.selection == 'median':
            idx = overall_ranking[len(overall_ranking) // 2 - 1] - 1
        else:
            assert self.selection == 'fast', f'invalid selection: {self.selection}'
            idx = overall_ranking[0] - 1

        return self._make_graph_stat_double(graph=generated_graphs[idx], idx=idx, scores=scores)

    def run(self, use_pickle: bool) -> None:
        """
        New runner - don't expand into three - grow three separate branches
        :param use_pickle:
        :return:
        """
        pickle_ext = '.pkl.gz'
        if use_pickle and check_file_exists(self.root_pickle_path + pickle_ext):
            CP.print_green(f'Using pickle at "{self.root_pickle_path + pickle_ext}"')
            self.root = load_pickle(self.root_pickle_path + pickle_ext)
            return

        tqdm.write(
            f'Running Infinity Mirror on "{self.initial_graph.name}" {self.initial_graph.order(), self.initial_graph.size()} "{self.model.model_name}" {self.num_generations} generations')
        pbar = tqdm(total=self.num_generations, bar_format='{l_bar}{bar}|[{elapsed}<{remaining}]', ncols=50)

        for i in range(self.num_generations):
            if i == 0:
                tnode = self.root  # start with the root
                curr_graph = self.initial_graph  # current graph is the initial graph

            level = i + 1
            self.model.update(new_input_graph=curr_graph)  # update the model
            generated_graphs = self.model.generate(num_graphs=self.num_graphs,
                                                   gen_id=level)  # generate a new set of graphs

            graph_stats = self._get_representative_graph_stat(generated_graphs=generated_graphs)

            if graph_stats is None:
                CP.print_blue('Infinity mirror failed')
                self.root_pickle_path += f'_failed-{level}'  # append the failed to filename
                break

            curr_graph, stats = graph_stats
            curr_graph.name = f'{self.initial_graph.name}_{self.selection}_{level}_{self.run_id}'
            tnode = TreeNode(name=f'{self.selection}_{level}', graph=curr_graph, stats=stats, parent=tnode)
            pbar.update(1)

        pbar.close()
        CP.print_green(f'Root object is pickled at "{self.root_pickle_path + pickle_ext}"')
        pickle.dump(self.root, open(self.root_pickle_path + pickle_ext, 'wb'), protocol=-1)  # use highest possible protocol
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
                             'sel': self.selection, 'gens': self.num_generations, 'time': time_taken})

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
                             'sel': self.selection, 'gens': self.num_generations, 'level': level})

        return
