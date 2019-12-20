import networkx as nx

from collections import defaultdict
from typing import Any, List, Dict
import matplotlib.pyplot as plt
from matplotlib import gridspec
import math

from src.graph_models import *
from src.Graph import CustomGraph
from src.graph_io import GraphReader


class InfinityMirror:
    def __init__(self, initial_graph: CustomGraph, model_obj: Any, num_generations: int):
        self.initial_graph: CustomGraph = initial_graph  # the initial starting point H_0
        self.num_generations: int = num_generations  # number of generations
        self.model: BaseGraphModel = self.init_model(model_obj)  # init BaseGraphModel object based on the parameters
        self.current_generation: int = 0  # counter for current generations
        self.graphs_by_generation: Dict[int, List[CustomGraph]] = {0: [self.initial_graph]}

    def __str__(self):
        return f'model: "{self.model.model_name}"  initial graph: "{self.initial_graph.name}"  #gens: {self.num_generations}'

    def __repr__(self):
        return str(self)

    def init_model(self, model_obj) -> BaseGraphModel:
        """
        Initializes Infinity Mirror object - calls the right Model object
        """
        return model_obj(input_graph=self.initial_graph)  # initialize and fit the model

    def _get_next_generation(self, num_graphs: int=1) -> None:
        """
        step 1: get input graph
        step 2: fit model
        step 3: generate output graphs - best and worst?
        step 4: fit output graph as input
        :param input_graph:
        :param num_graphs: number of graphs to generate
        :return:
        """
        input_graph: CustomGraph = self.graphs_by_generation[self.current_generation][0]  # use the prior generation's graph as input

        self.current_generation += 1  # update current generation
        self.model.update(new_input_graph=input_graph)
        self.model.generate(num_graphs=num_graphs, gen_id=self.current_generation)  # populates self.generated_graphs list
        self.graphs_by_generation[self.current_generation] = self.model.generated_graphs

        return

    def run(self, num_graphs: int=1) -> None:
        for _ in range(self.num_generations):
            self._get_next_generation(num_graphs=num_graphs)
        return

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
            ax = fig.add_subplot(gs[i])
            graph.plot(ax=ax, pos=pos, update_pos=False)

        fig.tight_layout()
        fig.suptitle(self.model.model_name)
        plt.grid(False)
        # plt.title(self.model.model_name)
        plt.show()


def main():
    # g = nx.path_graph(20)
    # g = CustomGraph(g, gen_id=0)
    # g.name = f'ladder_graph'

    graph_reader = GraphReader(filename='../input/karate.g', reindex_nodes=True, first_label=0)
    g = graph_reader.graph

    inf = InfinityMirror(initial_graph=g, num_generations=5, model_obj=CNRG)

    # inf.graphs_by_generation[0][0].plot(prog='neato')
    # plt.style.use('seaborn-white')
    # plt.grid(False)
    # plt.show()
    print(inf)


if __name__ == '__main__':
    main()
