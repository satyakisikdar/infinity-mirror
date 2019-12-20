"""
Graph Comparison Functions
"""
from typing import Dict

import networkx as nx
import numpy as np
from numpy import linalg as la

from src.utils import fast_bp, _pad
from src.Graph import CustomGraph


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

        return lambda_d

    def deltacon0(self, eps=None):
        n1, n2 = self.graph1.order(), self.graph2.order()
        N = max(n1, n2)

        A1, A2 = [_pad(A, N) for A in [nx.to_numpy_array(self.graph1), nx.to_numpy_array(self.graph2)]]
        S1, S2 = [fast_bp(A, eps=eps) for A in [A1, A2]]
        dist = np.abs(np.sqrt(S1) - np.sqrt(S2)).sum()

        self.stats['deltacon0'] = round(dist, 3)

        return round(dist, 3)

# def compare_two_graphs(g_true: CustomGraph, g_test: Union[CustomGraph, LightMultiGraph], true_deg=None, true_page=None):
#     """
#     Compares two graphs
#     :param g_true: actual graph
#     :param g_test: generated graph
#     :return:
#     """
#     if true_deg is None:
#         true_deg = nx.degree_histogram(g_true)
#
#     if true_page is None:
#         true_page = list(nx.pagerank_scipy(g_true).values())
#
#     start = time()
#     g_test_deg = nx.degree_histogram(g_test)
#     deg_time = time() - start
#
#     start = time()
#     g_test_pr = list(nx.pagerank_scipy(g_test).values())
#     page_time = time() - start
#
#     start = time()
#     gcd = GCD(g_true, g_test, 'orca')
#     gcd_time = time() - start
#
#     start = time()
#     cvm_deg = cvm_distance(true_deg, g_test_deg)
#     cvm_page = cvm_distance(true_page, g_test_pr)
#     cvm_time = time() - start
#
#     ld = lambda_dist(g_true, g_test, k=min(g_true.order(), g_test.order(), 10))
#
#     dc0 = deltacon0(g_true, g_test)
#
#     logging.debug(f'times: deg {round(deg_time, 3)}s, page {round(page_time, 3)}s, gcd {round(gcd_time, 3)}s, cvm {round(cvm_time, 3)}s')
#     return gcd, cvm_deg, cvm_page, ld, dc0
