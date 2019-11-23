'''
Container for different graph models
'''
import networkx as nx
from typing import List, Dict, Any

class BaseGraphModel:
    def __init__(self, model_name: str, input_graph: nx.Graph):
        self.input_graph = input_graph
        self.model_name = model_name
        self.params: Dict[Any] = {}  # dictionary of model parameters
        self.generated_graphs: List[nx.Graph] = []

    def fit(self):
        pass

    def generate_graphs(self, num_graphs: int):
        pass

    def __str__(self):
        return f'name: {self.model_name}, input_graph: {self.input_graph.name}, params: {self.params}'

    def __repr__(self):
        return str(self)

class ErdosRenyi(BaseGraphModel):
    def __init__(self, input_graph: nx.Graph):
        super().__init__(model_name='Erdos-Renyi', input_graph=input_graph)

    def fit(self):
        self.params = {}
        n = self.input_graph.order()
        m = self.input_graph.size()

        self.params['n'] = n
        self.params['prob'] = m / (n * (n - 1) / 2)


    def generate_graphs(self, num_graphs) -> List[nx.Graph]:
        graphs = []
        for _ in range(num_graphs):
            g = nx.erdos_renyi_graph(n=self.params['n'], p=self.params['prob'])
            graphs.append(g)
        return graphs