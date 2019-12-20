import networkx as nx

from typing import Any, List, Dict
from src.graph_models import *
from src.graph_models import BaseGraphModel
from src.Graph import CustomGraph

class InfinityMirror:
    def __init__(self, initial_graph: CustomGraph, model_obj: Any, num_generations: int):
        self.initial_graph = initial_graph  # the initial starting point H_0
        self.num_generations = num_generations  # number of generations
        self.model_obj = model_obj  # BaseGraphModel object
        self.model: BaseGraphModel = self.init_model()  # init BaseGraphModel object based on the parameters
        self.current_generation = 0  # counter for current generations
        self.graphs_by_generation: Dict[int, List[CustomGraph]] = {0: [self.initial_graph]}

    def __str__(self):
        return f'model: "{self.model.model_name}"  initial graph: "{self.initial_graph.name}"  #gens: {self.num_generations}'

    def __repr__(self):
        return str(self)

    def init_model(self) -> BaseGraphModel:
        """
        Initializes Infinity Mirror object - calls the right Model object
        """
        return self.model_obj(input_graph=self.initial_graph)  # initialize and fit the model

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

def main():
    g = nx.path_graph(10)
    g = CustomGraph(g, gen_id=0)
    g.name = f'path_graph'

    inf = InfinityMirror(initial_graph=g, num_generations=5, model_obj=ErdosRenyi)
    inf.run()
    print(inf)


if __name__ == '__main__':
    main()
