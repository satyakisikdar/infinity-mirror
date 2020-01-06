"""
Graph Comparison Functions
"""
from typing import Dict

import networkx as nx
import numpy as np
from numpy import linalg as la

from src.utils import fast_bp, _pad, cvm_distance
# from src.Graph import CustomGraph
from src.GCD import GCD
from src.graph_stats import GraphStats


class GraphPairCompare:
    """
    Compares two graphs
    """
    __slots__ = ['gstats1', 'gstats2', 'graph1', 'graph2', 'stats']

    def __init__(self, gstats1: GraphStats, gstats2: GraphStats) -> None:
        self.gstats1: GraphStats = gstats1
        self.gstats2: GraphStats = gstats2
        self.graph1: nx.Graph = gstats1.graph
        self.graph2: nx.Graph = gstats2.graph
        self.stats: Dict[str, float] = {}
        self.calculate()
        return

    def __str__(self) -> str:
        st = f'Comparing graphs: "{self.graph1.name}" and "{self.graph2.name}"'
        for key, val in self.stats:
            st += f'\n{key}: {round(val, 3)}'
        return st

    def __getitem__(self, item) -> float:
        assert item in self.stats, f'Invalid {item} in {self.stats.keys()}'
        return self.stats[item]

    def calculate(self) -> None:
        self.lambda_dist(k=10)
        self.gcd()
        self.deltacon0()
        self.cvm_degree()
        self.cvm_pagerank()
        return

    def gcd(self) -> float:
        dist = GCD(self.graph1, self.graph2)
        self.stats['gcd'] = dist

        return round(dist, 3)

    def lambda_dist(self, k=None, p=2) -> float:
        """
        compare the euclidean distance between the top-k eigenvalues of the laplacian
        :param k:
        :param p:
        :return:
        """
        if k is None:
            k = min(self.graph1.order(), self.graph2.order())
        else:
            k = min(self.graph1.order(), self.graph2.order(), k)

        lambda_seq_1 = np.array(sorted(self.gstats1['laplacian_eigenvalues'], reverse=True))[: k]
        lambda_seq_2 = np.array(sorted(self.gstats2['laplacian_eigenvalues'], reverse=True)[: k])

        dist = round(la.norm(lambda_seq_1 - lambda_seq_2, ord=p) / k, 3)
        self.stats['lambda_dist'] = dist

        return round(dist, 3)

    def deltacon0(self, eps=None) -> float:
        n1, n2 = self.graph1.order(), self.graph2.order()
        N = max(n1, n2)

        A1, A2 = [_pad(A, N) for A in [nx.to_numpy_array(self.graph1), nx.to_numpy_array(self.graph2)]]
        S1, S2 = [fast_bp(A, eps=eps) for A in [A1, A2]]
        dist = np.abs(np.sqrt(S1) - np.sqrt(S2)).sum()

        self.stats['deltacon0'] = round(dist, 3)

        return round(dist, 3)

    def cvm_pagerank(self) -> float:
        """
        Calculate the CVM distance of the pagerank
        """
        pr1 = list(self.gstats1['pagerank'].values())
        pr2 = list(self.gstats2['pagerank'].values())

        dist = cvm_distance(pr1, pr2)
        self.stats['pagerank_cvm'] = dist

        return round(dist, 3)

    def cvm_degree(self) -> float:
        """
        Calculate the CVM distance of the degree distr
        """
        # if deg1 is None:
        #     deg1 = nx.degree_histogram(self.graph1)
        # if deg2 is None:
        #     deg2 = nx.degree_histogram(self.graph2)
        deg1 = list(self.gstats1['degree_dist'].values())
        deg2 = list(self.gstats2['degree_dist'].values())

        dist = cvm_distance(deg1, deg2)
        self.stats['degree_cvm'] = dist

        return round(dist, 3)

