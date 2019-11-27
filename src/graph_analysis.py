"""
Container for different graph comparison metrics
"""
from typing import Dict, Tuple, List
import networkx as nx
from collections import Counter, deque
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("darkgrid")


class GraphStats:
    """
    GraphStats has methods for finding different statistics for a NetworkX graph
    """
    def __init__(self, graph: nx.Graph):
        self.graph = graph
        self.stats: Dict[str, float] = {}

    def diameter(self) -> float:
        return nx.diameter(self.g)

    def degree_dist(self) -> Dict[int, int]:
        """
        Returns the degrees counter - keys: degrees, values: #nodes with that degree
        :return:
        """
        degree_seq = sorted(deg for _, deg in self.graph.degree())
        degree_counts = Counter(degree_seq)
        return dict(degree_counts)

    def k_hop_reachability(self) -> np.array:
        """
        Returns the average number of nodes reachable from any node in k-hops
        Two levels of aggregation:
            1. _k_hop_reachability gives the absolute count of nodes reachable within a k-hops from a node
            2. overall_k_hop_dict aggregates the sum of all absolute counts for all nodes
        Normalizing factor: n ** 2 (once for each step)
        Then convert to a cumulative distribution
        :return:
        """
        overall_k_hop_dict = Counter()
        for node in self.graph.nodes():
            k_hop_dict = self._k_hop_reachability_counter(node)
            overall_k_hop_dict += Counter(k_hop_dict)

        k_hop_vec = np.array([v for k, v in sorted(overall_k_hop_dict.items(), key=lambda x: x[0])])
        k_hop_vec = k_hop_vec / (self.graph.order() ** 2)
        return np.cumsum(k_hop_vec)

    def _k_hop_reachability_counter(self, node) -> Dict[int, float]:
        """
        computes fraction of nodes reachable from the given node in k hops
        :param node: node to compute the k_hop_reachability vector
        :return:
        """
        n = self.graph.order()

        reachability_counter = {0: 1}  # within 0 hops, you can reach 1 node - itself
        hop_counter = {node: 0}  # node is 0 hops away from itself
        queue = deque([node])

        while len(queue) != 0:
            node = queue.popleft()
            for nbr in self.graph.neighbors(node):
                if nbr not in hop_counter:  # unexplored neighbor
                    hop_counter[nbr] = hop_counter[node] + 1  # update hop distance of neighbor

                    if hop_counter[nbr] not in reachability_counter:
                        reachability_counter[hop_counter[nbr]] = 0 # reachability_counter[hop_counter[node]]
                    reachability_counter[hop_counter[nbr]] += 1  # keep track of fraction of nodes reachable

                    queue.append(nbr)

        # normalized_reachability_counter = {key: value / n for key, value in reachability_counter.items()}
        return reachability_counter

    def effective_diameter(self) -> None:
        """
        Returns the 90% effective diameter of a graph
        :return:
        """
        raise (NotImplementedError)

    def pgd_graphlet_counts(self) -> None:
        """
        Return the dictionary of graphlets and their counts - based on Neville's PGD
        :return:
        """
        raise (NotImplementedError)

    def orca_graphlet_counts(self) -> None:
        """
        Return the dictionary of graphlets as counted by Orca
        :return:
        """
        raise (NotImplementedError)

def make_plot(y, kind='line', x=None, title='', xlabel='', ylabel='') -> None:
    if isinstance(x, dict):
        x, y = zip(*x)
    else: # if isinstance(x, list) or isinstance(x, np.array):
        x = list(range(len(y)))

    if kind == 'line':
        plt.plot(x, y, marker='o', linestyle='--')
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.show()
    return
