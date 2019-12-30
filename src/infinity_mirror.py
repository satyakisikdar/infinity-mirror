import math
from collections import defaultdict, namedtuple
from statistics import median_low  # median_low returns the lower median
from typing import Any, List, Dict

import matplotlib.pyplot as plt
import networkx as nx
from matplotlib import gridspec

from src.Graph import CustomGraph
from src.graph_comparison import GraphPairCompare
from src.graph_models import *
from src.graph_stats import GraphStats
from src.utils import borda_sort

# mpl.rcParams['figure.dpi'] = 600

Stats = namedtuple('Stats', 'id graph score')
GraphTriple = namedtuple('GraphTriple', 'best worst median')

class InfinityMirror:
    """
    Class for InfinityMirror
    For each generation, store 3 graphs - best, worst, and the 50^th percentile -
        use ranked choice voting for deciding the winner from 10 graphs <- rcv gives the

    """
    __slots__ = ('initial_graph', 'num_generations', 'model', '_current_generation', 'graphs_by_generation',
                'filtered_graphs_by_generation', 'initial_graph_stats')

    def __init__(self, initial_graph: CustomGraph, model_obj: Any, num_generations: int) -> None:
        self.initial_graph: CustomGraph = initial_graph  # the initial starting point H_0
        self.num_generations: int = num_generations  # number of generations
        self.model: BaseGraphModel = self.init_model(model_obj)  # init BaseGraphModel object based on the parameters
        self._current_generation: int = 0  # counter for current generations
        self.graphs_by_generation: Dict[int, List[CustomGraph]] = {0: [self.initial_graph]}  # stores ALL the graphs for every generation
        self.filtered_graphs_by_generation: Dict[int, GraphTriple] = {0: GraphTriple(best=self.initial_graph, worst=None, median=None)}  # stores only 3 graphs per generation
        self.initial_graph_stats = GraphStats(graph=self.initial_graph)  # initialize graph_stats object for the initial_graph which is the same across generations
        return

    def __str__(self):
        return f'model: "{self.model.model_name}"  initial graph: "{self.initial_graph.name}"  #gens: {self.num_generations}'

    def __repr__(self):
        return str(self)

    def init_model(self, model_obj) -> BaseGraphModel:
        """
        Initializes Infinity Mirror object - calls the right Model object
        """
        return model_obj(input_graph=self.initial_graph)  # initialize and fit the model

    def _get_next_generation(self, input_graph: CustomGraph, num_graphs: int) -> None:
        """
        step 1: get input graph
        step 2: fit model
        step 3: generate output graphs - best and worst?
        step 4: fit output graph as input
        :param input_graph:
        :param num_graphs: number of graphs to generate
        :return:
        """
        ## set input_graph to use the 3 chosen graphs not just the one graph
        # input_graph: CustomGraph = self.graphs_by_generation[self.current_generation][0]  # use the prior generation's graph as input

        self._current_generation += 1  # update current generation
        self.model.update(new_input_graph=input_graph)
        self.model.generate(num_graphs=num_graphs, gen_id=self._current_generation)  # populates self.generated_graphs list
        self.graphs_by_generation[self._current_generation] = self.model.generated_graphs
        self.filtered_graphs_by_generation[self._current_generation] = self._filter_graphs()

        return

    def _filter_graphs(self) -> None:
        """
        Filter the graphs per generation to store the 3 chosen graphs - best, worst, and 50^th percentile
        Populates the filtered_graphs_by_generation
        :return: None
        """
        assert self._current_generation in self.graphs_by_generation, f'Invalid generation {self._current_generation}'
        assert len(self.graphs_by_generation[self._current_generation]) != 0, f'Graph list empty for gen: {self._current_generation}'

        ## For each graph in the generation
        #### compute graph distance with the original input graph and the generated graph
        #### create a ranked list based on the scores
        ## combine the ranked lists to create an overall ranking
        ## pick the best, worst, and the median - use named tuple?

        scores: Dict[str, List[Stats]] = {'gcd': [], 'deltacon0': [], 'pagerank_cvm': [], 'degree_cvm': [], 'lambda_dist': []}

        gen_graphs = self.graphs_by_generation[self._current_generation]
        for i, gen_graph in enumerate(gen_graphs):
            gen_gstats = GraphStats(gen_graph)
            graph_comp = GraphPairCompare(gstats1=self.initial_graph_stats, gstats2=gen_gstats)
            for metric in scores.keys():
                stat = Stats(id=i+1, graph=gen_graph, score=graph_comp[metric])
                scores[metric].append(stat)

        sorted_scores = {key: sorted(val, key=lambda item: item.score) for key, val in scores.items()}

        rankings: Dict[str, List[int]] = {}  # stores the id of the graphs sorted by score
        for metric, stats in sorted_scores.items():
            rankings[metric] = list(map(lambda item: item.id, stats))

        overall_ranking = borda_sort(rankings.values())
        best_graph = gen_graphs[overall_ranking[0] - 1]  # ranking is 1-based
        worst_graph = gen_graphs[overall_ranking[-1] - 1]  # same
        median_graph = gen_graphs[median_low(overall_ranking) - 1]
        self.filtered_graphs_by_generation[self._current_generation] = GraphTriple(best=best_graph, worst=worst_graph, median=median_graph)
        return

    def run(self, num_graphs: int=10) -> None:
        ## start with the input graph, but moving forward, use the best, worst, and median graphs as seeds from the filtered_graphs list -
        ## each spawning 3 graphs each - keep a tree of the graphs

        if self._current_generation == 0:
            input_graph = self.initial_graph  # start with the initial graph


        # self._get_next_generation(num_graphs=num_graphs, input_graph=self.initial_graph)
        # for _ in range(self.num_generations):
        #     self._get_next_generation(num_graphs=num_graphs)
        # return
        pass

    def plot(self, prog: str = 'neato'):
        """
        Plot the progression of the infinity mirror - fix the node positions
        :return:
        """
        pos = defaultdict(lambda: (0, 0))  # the default dict
        pos.update(nx.nx_agraph.graphviz_layout(self.initial_graph, prog=prog))  # get the pos of nodes of the original nodes

        N = self.num_generations
        cols = 2
        rows = int(math.ceil(N / cols))

        gs = gridspec.GridSpec(rows, cols)
        fig = plt.figure()

        for i in range(N):
            graph: CustomGraph = self.graphs_by_generation[i][0]  # pick the 1st graph by default
            # gstats = GraphStats(graph)

            ax = fig.add_subplot(gs[i])
            graph.plot(ax=ax, pos=pos, update_pos=True)

            # deg_dist = gstats.degree_dist(normalized=True)
            # gstats.plot(y=deg_dist, title=f'Degree-Dist for {graph.name}', xlabel='Degree $k$', ylabel='Count of nodes',
            #           kind='scatter', ax=ax)

            # k_hop = gstats.k_hop_reach()
            # gstats.plot(y=k_hop, title=f'Hop-Plot for {graph.name}, gen:{i}, n:{graph.order():_}, m:{graph.size():_}',
            #             xlabel='Hops',) # ylabel='Avg. fraction of reachable nodes')

            # cc_by_deg = gstats.clustering_coefficients_by_degree()
            # gstats.plot(y=cc_by_deg, title=f'gen:{i}, n:{graph.order():_}, m:{graph.size():_}', # Avg cc by Degree (k)', xlabel='Degree $k$',
            #           ylabel='Avg cc', kind='scatter', ax=ax)


        fig.tight_layout()
        fig.suptitle(f'{graph.name} {self.model.model_name}', y=1)
        # plt.grid(False)
        # plt.title(self.model.model_name)
        plt.show()


def main():
    # g = nx.path_graph(20)
    g = nx.ring_of_cliques(500, 4)
    g = CustomGraph(g, gen_id=0)
    g.name = f'ring_cliq_500_4'

    # graph_reader = GraphReader(filename='../input/karate.g', reindex_nodes=True, first_label=0)
    # g = graph_reader.graph

    inf = InfinityMirror(initial_graph=g, num_generations=5, model_obj=CNRG)

    # inf.graphs_by_generation[0][0].plot(prog='neato')
    # plt.style.use('seaborn-white')
    # plt.grid(False)
    # plt.show()
    print(inf)


if __name__ == '__main__':
    main()
