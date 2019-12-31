import math
from collections import defaultdict, namedtuple, deque
from typing import Any, List, Dict, Deque

import matplotlib.pyplot as plt
import networkx as nx
from matplotlib import gridspec

from src.Graph import CustomGraph
from src.graph_comparison import GraphPairCompare
from src.graph_models import *
from src.graph_stats import GraphStats
from src.utils import borda_sort
from src.Tree import TreeNode

# mpl.rcParams['figure.dpi'] = 600

Stats = namedtuple('Stats', 'id graph score')
GraphTriple = namedtuple('GraphTriple', 'best worst median')

## Add anytree to store the tree of graphs

class InfinityMirror:
    """
    Class for InfinityMirror
    For each generation, store 3 graphs - best, worst, and the 50^th percentile -
        use ranked choice voting for deciding the winner from 10 graphs <- borda list
        store the three graphs into a tree
    """
    __slots__ = ('initial_graph', 'num_generations', 'model', 'initial_graph_stats', 'root')

    def __init__(self, initial_graph: CustomGraph, model_obj: Any, num_generations: int) -> None:
        self.initial_graph: CustomGraph = initial_graph  # the initial starting point H_0
        self.num_generations: int = num_generations  # number of generations
        self.model: BaseGraphModel = model_obj(input_graph=self.initial_graph)  # initialize and fit the model
        self.initial_graph_stats = GraphStats(graph=self.initial_graph)  # initialize graph_stats object for the initial_graph which is the same across generations
        self.root = TreeNode('root', graph=self.initial_graph)  # root of the tree with the initial graph
        return

    def __str__(self) -> str:
        return f'model: "{self.model.model_name}"  initial graph: "{self.initial_graph.name}"  #gens: {self.num_generations}'

    def __repr__(self) -> str:
        return str(self)

    def run(self, num_graphs: int=10):
        """
        Do a BFS starting with the root
        :return:
        """
        stack: List[TreeNode] = [self.root]

        while len(stack) != 0:
            tnode = stack.pop()
            if tnode.depth >= self.num_generations:  # do not further expand the tree
                continue

            graph_triple = self._get_next_generation(input_graph=tnode.graph, num_graphs=num_graphs, gen_id=tnode.depth+1)
            best_graph, worst_graph, median_graph = graph_triple.best, graph_triple.worst, graph_triple.median

            ## creating the three nodes and attaching it to the tree
            TreeNode(name=f'{tnode}-best', graph=best_graph, parent=tnode)
            TreeNode(name=f'{tnode}-med', graph=median_graph, parent=tnode)
            TreeNode(name=f'{tnode}-worst', graph=worst_graph, parent=tnode)

            assert len(tnode.children) == 3, f'tree node {tnode} does not have 3 children'
            stack.extend(tnode.children)  # add the children to the end of the queue
        return

    def _get_next_generation(self, input_graph: CustomGraph, num_graphs: int, gen_id: int) -> GraphTriple:
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

    def _filter_graphs(self, generated_graphs: List[CustomGraph]) -> GraphTriple:
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

        scores: Dict[str, List[Stats]] = {'gcd': [], 'deltacon0': [], 'pagerank_cvm': [], 'degree_cvm': [], 'lambda_dist': []}

        for i, gen_graph in enumerate(generated_graphs):
            gen_gstats = GraphStats(gen_graph)
            graph_comp = GraphPairCompare(gstats1=self.initial_graph_stats, gstats2=gen_gstats)
            for metric in scores.keys():
                stat = Stats(id=i+1, graph=gen_graph, score=graph_comp[metric])
                scores[metric].append(stat)

        sorted_scores = {key: sorted(val, key=lambda item: item.score) for key, val in scores.items()}

        rankings: Dict[str, List[int]] = {}  # stores the id of the graphs sorted by score
        for metric, stats in sorted_scores.items():
            rankings[metric] = list(map(lambda item: item.id, stats))

        overall_ranking = borda_sort(rankings.values())  # compute the overall ranking

        best_graph = generated_graphs[overall_ranking[0] - 1]  # ranking is 1-based
        worst_graph = generated_graphs[overall_ranking[-1] - 1]  # same
        median_graph = generated_graphs[overall_ranking[len(overall_ranking)//2 - 1] - 1]   # 5th element from the list

        return GraphTriple(best=best_graph, worst=worst_graph, median=median_graph)

    def plot(self, prog: str='neato'):
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
