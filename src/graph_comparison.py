"""
Graph Comparison Functions
"""
from typing import Dict

import networkx as nx
import numpy as np
from numpy import linalg as la

from src.utils import fast_bp, _pad, cvm_distance
from src.Graph import CustomGraph
from src.GCD import GCD

class GraphPairCompare:
    """
    Compares two graphs
    """
    def __init__(self, graph1: CustomGraph, graph2: CustomGraph):
        self.graph1: CustomGraph = graph1
        self.graph2: CustomGraph = graph2
        self.stats: Dict[str, float] = {}

    def __str__(self) -> str:
        st = f'Comparing graphs: "{self.graph1.name}" and "{self.graph2.name}"'
        for key, val in self.stats:
            st += f'\n{key}: {round(val, 3)}'
        return st

    def __getitem__(self, item):
        return self.stats[item]

    def calculate(self):
        pass

    def gcd(self) -> float:
        return GCD(self.graph1, self.graph2)

    def lambda_dist(self, k=None, p=2) -> float:
        """
        compare the euclidean distance between the top-k eigenvalues of the laplacian
        :param k:
        :param p:
        :return:
        """
        if k is None:
            k = min(self.graph1.order(), self.graph2.order())

        lambda_seq_1 = np.array(sorted(nx.linalg.laplacian_spectrum(self.graph1), reverse=True)[: k])
        lambda_seq_2 = np.array(sorted(nx.linalg.laplacian_spectrum(self.graph2), reverse=True)[: k])

        lambda_d = round(la.norm(lambda_seq_1 - lambda_seq_2, ord=p) / k, 3)
        self.stats['lambda_dist'] = lambda_d

        return round(lambda_d, 3)

    def deltacon0(self, eps=None) -> float:
        n1, n2 = self.graph1.order(), self.graph2.order()
        N = max(n1, n2)

        A1, A2 = [_pad(A, N) for A in [nx.to_numpy_array(self.graph1), nx.to_numpy_array(self.graph2)]]
        S1, S2 = [fast_bp(A, eps=eps) for A in [A1, A2]]
        dist = np.abs(np.sqrt(S1) - np.sqrt(S2)).sum()

        self.stats['deltacon0'] = round(dist, 3)

        return round(dist, 3)

    def cvm_pagerank(self, pr1=None, pr2=None) -> float:
        """
        Calculate the CVM distance of the pagerank
        """
        if pr1 is None:
            pr1 = list(nx.pagerank_scipy(self.graph1).values())
        if pr2 is None:
            pr2 = list(nx.pagerank_scipy(self.graph2).values())

        dist = cvm_distance(pr1, pr2)
        self.stats['pagerank_cvm'] = dist
        return round(dist, 3)

    def cvm_degree(self, deg1=None, deg2=None) -> float:
        """
        Calculate the CVM distance of the degree distr
        """
        if deg1 is None:
            deg1 = nx.degree_histogram(self.graph1)
        if deg2 is None:
            deg2 = nx.degree_histogram(self.graph2)

        dist = cvm_distance(deg1, deg2)
        self.stats['degree_cvm'] = dist
        return round(dist, 3)
