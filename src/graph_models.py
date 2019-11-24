'''
Container for different graph models
'''
import networkx as nx
from typing import List, Dict, Any
import abc

class BaseGraphModel:
    def __init__(self, model_name: str, input_graph: nx.Graph):
        self.input_graph: nx.Graph = input_graph  # networkX graph to be fitted
        self.model_name = model_name  # name of the model
        self.params: Dict[Any] = {}  # dictionary of model parameters
        self.generated_graphs: List[nx.Graph] = []   # list of NetworkX graphs

    @abc.abstractmethod
    def fit(self):
        pass

    def generate_graphs(self, num_graphs: int):
        for _ in range(num_graphs):
            g = self.generate()
            self.generated_graphs.append(g)

    @abc.abstractmethod
    def generate(self) -> nx.Graph:
        """
        Generates one graph
        """
        pass

    def __str__(self):
        return f'name: {self.model_name}, input_graph: {self.input_graph.name}, params: {self.params}'

    def __repr__(self):
        return str(self)


class ErdosRenyi(BaseGraphModel):
    def __init__(self, input_graph: nx.Graph):
        super().__init__(model_name='Erdos-Renyi', input_graph=input_graph)

    def fit(self):
        n = self.input_graph.order()
        m = self.input_graph.size()

        self.params['n'] = n
        self.params['p'] = m / (n * (n - 1) / 2)

    def generate(self) -> nx.Graph:
        assert 'n' in self.params and 'p' in self.params, 'Improper parameters for Erdos-Renyi'
        return nx.erdos_renyi_graph(n=self.params['n'], p=self.params['p'])


class ChungLu(BaseGraphModel):
    def __init__(self, input_graph: nx.Graph):
        super().__init__(model_name='Chung-Lu', input_graph=input_graph)

    def fit(self):
        self.params['degree_seq'] = sorted(deg for node, deg in self.input_graph.degree())

    def generate(self) -> nx.Graph:
        assert 'degree_seq' in self.params, 'imporper parameters for Chung-Lu'

        g = nx.configuration_model(self.params['degree_seq'])  # fit the model to the degree seq
        g = nx.Graph(g)  # make it into a simple graph
        g.remove_edges_from(nx.selfloop_edges(g))  # remove self-loops

        return g
