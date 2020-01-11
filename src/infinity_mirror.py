import math
from collections import defaultdict, namedtuple
from typing import Any, List, Dict

import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
import pickle
import numpy as np
from tqdm import tqdm
from joblib import Parallel, delayed
from matplotlib import gridspec

from src.graph_comparison import GraphPairCompare
from src.graph_models import *
from src.graph_stats import GraphStats
from src.utils import borda_sort, mean_confidence_interval, ColorPrint as CP, load_pickle, check_file_exists
from src.Tree import TreeNode

# mpl.rcParams['figure.dpi'] = 600

Stats = namedtuple('Stats', 'name id graph score')  # stores the different stats for each graph. name: name of metric, id: graph_id
GraphStatDouble = namedtuple('GraphStatDouble', 'graph stats')   # graph: NetworkX object, stat: dictionary of comparison stats with the input
GraphStatTriple = namedtuple('GraphStatTriple', 'best worst median')  # stores the best, worst, and median graphs and their stats (GraphStat double)


# TODO: write new plotting method to plot the confidence intervals across generations

class InfinityMirror:
    """
    Class for InfinityMirror
    For each generation, store 3 graphs - best, worst, and the 50^th percentile -
        use ranked choice voting for deciding the winner from 10 graphs <- borda list
        store the three graphs into a tree
    """
    __slots__ = ('initial_graph', 'num_generations', 'num_graphs', 'model', 'initial_graph_stats', 'root',
                 '_metrics', 'root_pickle_path', 'selection')

    def __init__(self, selection: str, initial_graph: nx.Graph, model_obj: Any, num_generations: int, num_graphs: int) -> None:
        self.selection = selection  # kind of selection stategy
        self.initial_graph: nx.Graph = initial_graph  # the initial starting point H_0
        self.num_graphs: int = num_graphs  # number of graphs per generation
        self.num_generations: int = num_generations  # number of generations
        self.model: BaseGraphModel = model_obj(input_graph=self.initial_graph)  # initialize and fit the model
        self.initial_graph_stats: GraphStats = GraphStats(graph=self.initial_graph)  # initialize graph_stats object for the initial_graph which is the same across generations
        self.root: TreeNode = TreeNode('root', graph=self.initial_graph, stats={})  # root of the tree with the initial graph and empty stats dictionary
        self._metrics: List[str] = ['gcd', 'deltacon0', 'lambda_dist', 'pagerank_cvm', 'degree_cvm']  # list of metrics
        self.root_pickle_path: str = f'./output/pickles/{self.initial_graph.name}/{self.selection}_{self.model.model_name}_{self.num_generations}.pkl.gz'
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
        stats = {metric: scores[metric][idx].score for metric in self._metrics}
        return GraphStatDouble(graph=graph, stats=stats)

    def _get_representative_graph_stat(self, generated_graphs: List[nx.Graph]) -> GraphStatDouble:
        """
        returns the representative graph and its stats
        :param kind: str: best, median, worst
        :param generated_graphs: list of generated graphs
        :return: GraphStatDouble object
        """
        assert len(generated_graphs) != 0, f'generated graphs empty'
        scores: Dict[str, List[Stats]] = {metric: [] for metric in self._metrics}

        graph_comps_list = Parallel()(
            delayed(GraphPairCompare)(gstats1=self.initial_graph_stats, gstats2=GraphStats(gen_graph))
            for i, gen_graph in enumerate(generated_graphs))

        assert isinstance(graph_comps_list, list), 'Graph comp pairs is not a list'
        assert isinstance(graph_comps_list[0], GraphPairCompare), 'Improper object in Graph comp list'

        for i, graph_comp in enumerate(graph_comps_list):
            graph_comp: GraphPairCompare
            for metric in self._metrics:
                scores[metric].append(Stats(id=i + 1, graph=graph_comp.graph2, score=graph_comp[metric], name=metric))

        sorted_scores = {key: sorted(val, key=lambda item: item.score) for key, val in scores.items()}

        rankings: Dict[str, List[int]] = {}  # stores the id of the graphs sorted by score
        for metric, stats in sorted_scores.items():
            rankings[metric] = list(map(lambda item: item.id, stats))

        overall_ranking = borda_sort(rankings.values())  # compute the overall ranking

        if self.selection == 'best':
            idx = overall_ranking[0] - 1  # all indexing is 1 based
        elif self.selection == 'worst':
            idx = overall_ranking[-1] - 1
        else:
            assert self.selection == 'median', f'invalid selection: {self.selection}'
            idx = overall_ranking[len(overall_ranking) // 2 - 1] - 1

        return self._make_graph_stat_double(graph=generated_graphs[idx], idx=idx, scores=scores)

    def run(self, use_pickle: bool) -> None:
        """
        New runner - don't expand into three - grow three separate branches
        :param use_pickle:
        :return:
        """
        if use_pickle and check_file_exists(self.root_pickle_path):
            CP.print_green(f'Using pickle at "{self.root_pickle_path}"')
            self.root = load_pickle(self.root_pickle_path)
            return

        tqdm.write(f'Running Infinity Mirror on "{self.initial_graph.name}" {self.initial_graph.order(), self.initial_graph.size()} "{self.model.model_name}" {self.num_generations} generations')
        pbar = tqdm(total=self.num_generations, bar_format='{l_bar}{bar}|[{elapsed}<{remaining}]', ncols=50)

        for i in range(self.num_generations):
            if i == 0:
                tnode = self.root  # start with the root
                curr_graph = self.initial_graph  # current graph is the initial graph

            level = i + 1
            self.model.update(new_input_graph=curr_graph)  # update the model
            generated_graphs = self.model.generate(num_graphs=self.num_graphs, gen_id=level)  # generate a new set of graphs
            curr_graph, stats = self._get_representative_graph_stat(generated_graphs=generated_graphs)
            tnode = TreeNode(name=f'{self.selection}_{level}', graph=curr_graph, stats=stats, parent=tnode)
            pbar.update(1)

        pbar.close()
        CP.print_green(f'Root object is pickled at "{self.root_pickle_path}"')
        pickle.dump(self.root, open(self.root_pickle_path, 'wb'))
        return

    def _group_by_gen(self, tnode: TreeNode) -> Dict:
        """
        Group the stats by descendants of tnode into best, median, and worst
        :param tnode:
        :param kind:
        :return:
        """
        agg_stats_by_gen = {}

        for metric in self._metrics:
            stats = tnode.stats
            agg_stats_by_gen[metric] = {1: [stats[metric]]}  # populate this for the tnode -> gen 1
            for gen in range(2, self.num_generations+1):
                agg_stats_by_gen[metric][gen] = []

        for desc in tnode.descendants:
            desc: TreeNode
            gen = desc.depth
            for metric in self._metrics:
                agg_stats_by_gen[metric][gen].append(desc.stats[metric])

        return agg_stats_by_gen

    def aggregate_stats(self) -> Dict:
        """
        group the descendants of root-best, root-median, root-worst
        :return:
        """
        root_best, root_med, root_worst = self.root.children
        aggregated_stats = {'best': self._group_by_gen(root_best), 'median': self._group_by_gen(root_med),
                           'worst': self._group_by_gen(root_worst)}

        return aggregated_stats

    def plot(self):
        """
        Plot the progression of the infinity mirror - for different metrics across generations
        from the aggregated stats, find mean and plot it then add 95% conf intervals
        :return:
        """
        aggregated_stats = self.aggregate_stats()
        compressed_stats_mean = {}  # with the mean
        compressed_stats_intervals = {}  # with the confidence intervals

        self._metrics = self._metrics[-2: ]  # use only the first two metrics

        for kind in ('best', 'median', 'worst'):
            compressed_stats_mean[kind] = {}
            compressed_stats_intervals[kind] = {}
            for metric in self._metrics:
                compressed_stats_mean[kind][metric] = list(map(lambda l: np.mean(l),
                                                               aggregated_stats[kind][metric].values()))
                compressed_stats_intervals[kind][metric] = list(map(lambda l: mean_confidence_interval(l),
                                                               aggregated_stats[kind][metric].values()))

        x = list(range(1, self.num_generations+1))

        rows = len(self._metrics)  # for each of the metrics
        cols = 1

        gs = gridspec.GridSpec(rows, cols)
        fig = plt.figure()

        for i, grid in enumerate(gs):
            ax = fig.add_subplot(grid)
            for kind in ('best', 'median', 'worst'):
                metric = self._metrics[i]
                ax1 = sns.lineplot(x, compressed_stats_mean[kind][metric], marker='o', alpha=0.75, label=kind, ax=ax)
                ax1.lines[-1].set_linestyle('--')
                if i != 0:  # disable legend on all the plots except the first
                    ax.get_legend().set_visible(False)

                plt.ylabel(f'{metric}')
                plt.xticks(x, x)

        plt.suptitle(f'Metrics across generations for {self.model.model_name}')
        plt.show()

