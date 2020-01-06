import math
from collections import defaultdict, namedtuple
from typing import Any, List, Dict

import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
import pickle
import numpy as np
import tqdm
from matplotlib import gridspec

# from src.Graph import CustomGraph
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
    __slots__ = ('initial_graph', 'num_generations', 'model', 'initial_graph_stats', 'root', '_metrics', 'root_pickle_path')

    def __init__(self, initial_graph: nx.Graph, model_obj: Any, num_generations: int) -> None:
        self.initial_graph: nx.Graph = initial_graph  # the initial starting point H_0
        self.num_generations: int = num_generations  # number of generations
        self.model: BaseGraphModel = model_obj(input_graph=self.initial_graph)  # initialize and fit the model
        self.initial_graph_stats: GraphStats = GraphStats(graph=self.initial_graph)  # initialize graph_stats object for the initial_graph which is the same across generations
        self.root: TreeNode = TreeNode('root', graph=self.initial_graph, stats={})  # root of the tree with the initial graph and empty stats dictionary
        self._metrics: List[str] = ['gcd', 'deltacon0', 'lambda_dist', 'pagerank_cvm', 'degree_cvm']  # list of metrics
        self.root_pickle_path: str = f'./output/pickles/{self.initial_graph.name}_{self.model.model_name}_{self.num_generations}.pkl.gz'
        return

    def __str__(self) -> str:
        return f'model: "{self.model.model_name}"  initial graph: "{self.initial_graph.name}"  #gens: {self.num_generations}'

    def __repr__(self) -> str:
        return str(self)

    def run(self, use_pickle: bool, num_graphs: int=10):
        """
        Do a BFS starting with the root
        :return:
        """

        if use_pickle and check_file_exists(self.root_pickle_path):
            CP.print_green(f'Using pickle at "{self.root_pickle_path}"')
            self.root = load_pickle(self.root_pickle_path)
            return

        stack: List[TreeNode] = [self.root]

        max_num_nodes = (3 ** (self.num_generations+1) - 1) / 2  # total number of nodes in the tree

        tqdm.tqdm.write(f'Running Infinity Mirror on "{self.initial_graph.name}" {self.initial_graph.order(), self.initial_graph.size()} "{self.model.model_name}" {self.num_generations} generations')
        pbar = tqdm.tqdm(total=max_num_nodes, bar_format='{l_bar}{bar}|[{elapsed}<{remaining}]', ncols=50)
        pbar.update(1)

        while len(stack) != 0:
            tnode = stack.pop()
            if tnode.depth >= self.num_generations:  # do not further expand the tree
                continue

            graph_stat_triple: GraphStatTriple = self._get_next_generation(input_graph=tnode.graph, num_graphs=num_graphs, gen_id=tnode.depth+1)
            best_graph_stat_double: GraphStatDouble = graph_stat_triple.best
            median_graph_stat_double: GraphStatDouble = graph_stat_triple.median
            worst_graph_stat_double: GraphStatDouble = graph_stat_triple.worst

            ## creating the three nodes and attaching it to the tree
            TreeNode(name=f'{tnode}-best', graph=best_graph_stat_double.graph, stats=best_graph_stat_double.stats, parent=tnode)
            TreeNode(name=f'{tnode}-med', graph=median_graph_stat_double.graph, stats=median_graph_stat_double.stats, parent=tnode)
            TreeNode(name=f'{tnode}-worst', graph=worst_graph_stat_double.graph, stats=worst_graph_stat_double.stats, parent=tnode)

            assert len(tnode.children) == 3, f'tree node {tnode} does not have 3 children'
            stack.extend(tnode.children)  # add the children to the end of the queue
            pbar.update(3)

        pbar.close()
        ## pickle the root
        CP.print_green(f'Root object is pickled at "{self.root_pickle_path}"')
        pickle.dump(self.root, open(self.root_pickle_path, 'wb'))

        return

    def _get_next_generation(self, input_graph: nx.Graph, num_graphs: int, gen_id: int) -> GraphStatTriple:
        """
        step 1: get input graph
        step 2: fit model
        step 3: generate output graphs - best and worst?
        step 4: fit output graph as input
        :param input_graph:
        :param num_graphs: number of graphs to generate
        :return:
        """
        # raise NotImplementedError('dont use current gen, keep update, generate; fix graphs_by_gen; fix filter graphs; ')

        self.model.update(new_input_graph=input_graph)
        generated_graphs = self.model.generate(num_graphs=num_graphs, gen_id=gen_id)
        return self._filter_graphs(generated_graphs)

    def _make_graph_stat_double(self, graph, scores, idx) -> GraphStatDouble:
        """
        Makes GraphStatDouble objects
        :param scores:
        :param idx:
        :return:
        """
        stats = {metric: scores[metric][idx].score for metric in self._metrics}
        return GraphStatDouble(graph=graph, stats=stats)

    def _filter_graphs(self, generated_graphs: List[nx.Graph]) -> GraphStatTriple:
        """
        Filter the graphs per generation to store the 3 chosen graphs - best, worst, and 50^th percentile

        Populates the filtered_graphs_by_generation
        :return: None
        """
        assert len(generated_graphs) != 0, f'generated graphs empty'

        ## For each graph in the generation
        #### compute graph distance with the original input graph and the generated graph
        #### create a ranked list based on the scores
        ## combine the ranked lists to create an overall ranking
        ## pick the best, worst, and the median - use named tuple?

        scores: Dict[str, List[Stats]] = {metric: [] for metric in self._metrics}

        ## TODO: add tqdm status bar for progress

        for i, gen_graph in enumerate(generated_graphs):
            gen_gstats = GraphStats(gen_graph)
            graph_comp = GraphPairCompare(gstats1=self.initial_graph_stats, gstats2=gen_gstats)
            for metric in self._metrics:
                stat = Stats(id=i+1, graph=gen_graph, score=graph_comp[metric], name=metric)
                scores[metric].append(stat)

        sorted_scores = {key: sorted(val, key=lambda item: item.score) for key, val in scores.items()}

        rankings: Dict[str, List[int]] = {}  # stores the id of the graphs sorted by score
        for metric, stats in sorted_scores.items():
            rankings[metric] = list(map(lambda item: item.id, stats))

        overall_ranking = borda_sort(rankings.values())  # compute the overall ranking

        best_idx = overall_ranking[0] - 1   # all indexing is 1 based
        median_idx = overall_ranking[len(overall_ranking)//2 - 1] - 1
        worst_idx = overall_ranking[-1] - 1

        best_graph_stat_double = self._make_graph_stat_double(graph=generated_graphs[best_idx], idx=best_idx, scores=scores)
        median_graph_stat_double = self._make_graph_stat_double(graph=generated_graphs[median_idx], idx=median_idx, scores=scores)
        worst_graph_stat_double = self._make_graph_stat_double(graph=generated_graphs[worst_idx], idx=worst_idx, scores=scores)

        return GraphStatTriple(best=best_graph_stat_double, median=median_graph_stat_double, worst=worst_graph_stat_double)

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

