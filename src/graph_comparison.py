"""
Graph Comparison Functions
"""
from math import fabs
from typing import Dict, List, Any

import networkx as nx
import numpy as np
import scipy.stats
from scipy.special import kl_div
from numpy import linalg as la

from src.utils import fast_bp, _pad, cvm_distance, ks_distance
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
        for key, val in self.stats.items():
            st += f'\n{key}: {round(val, 3)}'
        return st

    def __getitem__(self, item) -> float:
        assert item in self.stats, f'Invalid {item} in {self.stats.keys()}'
        return self.stats[item]

    def calculate(self) -> None:
        self.lambda_dist(k=10)
        self.node_diff()
        self.edge_diff()
        self.pgd_pearson()
        self.pgd_spearman()
        self.deltacon0()
        self.cvm_degree()
        self.cvm_pagerank()

        if self.graph2.order() == 1 or 'blank' in self.graph2.name:  # empty graph
            for key in self.stats:
                self.stats[key] = float('inf')  # set the distances to infinity
        return

    def pgd_spearman(self) -> float:
        graphlet_dict_1 = self.gstats1['pgd_graphlet_counts']
        graphlet_dict_2 = self.gstats2['pgd_graphlet_counts']

        if len(graphlet_dict_1) == 0 or len(graphlet_dict_2) == 0:
            dist = float('inf')
            self.stats['pgd_spearman'] = dist
            return dist

        sorted_counts_1: List[int] = list(
            map(lambda item: item[1], graphlet_dict_1.items()))  # graphlet counts sorted by graphlet name

        sorted_counts_2: List[int] = list(
            map(lambda item: item[1], graphlet_dict_2.items()))  # graphlet counts sorted by graphlet name

        dist = 1 - scipy.stats.spearmanr(sorted_counts_1, sorted_counts_2)[0]
        self.stats['pgd_spearman'] = dist

        return round(dist, 3)


    def pgd_pearson(self) -> float:
        graphlet_dict_1 = self.gstats1['pgd_graphlet_counts']
        graphlet_dict_2 = self.gstats2['pgd_graphlet_counts']

        if len(graphlet_dict_1) == 0 or len(graphlet_dict_2) == 0:
            dist = float('inf')
            self.stats['pgd_pearson'] = dist
            return dist

        sorted_counts_1 = list(
            map(lambda item: item[1], graphlet_dict_1.items()))  # graphlet counts sorted by graphlet name

        sorted_counts_2 = list(
            map(lambda item: item[1], graphlet_dict_2.items()))  # graphlet counts sorted by graphlet name

        dist = 1 - scipy.stats.pearsonr(sorted_counts_1, sorted_counts_2)[0]
        self.stats['pgd_pearson'] = dist

        return round(dist, 3)

    def node_diff(self) -> float:
        """

        :return:
        """
        dist = fabs(self.graph1.order() - self.graph2.order())
        self.stats['node_diff'] = dist

        return dist

    def edge_diff(self) -> float:
        """

        :return:
        """
        dist = fabs(self.graph1.size() - self.graph2.size())
        self.stats['edge_diff'] = dist

        return dist

    def gcd(self) -> float:
        """

        :return:
        """
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

    def ks_test(self) -> float:
        """
        Calculate the KS distance of the degree distr
        """
        deg1 = list(self.gstats1['degree_dist'].values())
        deg2 = list(self.gstats2['degree_dist'].values())

        dist = ks_distance(deg1, deg2)
        self.stats['ks_dist'] = dist

        return round(dist, 3)

    def kl_divergence(self) -> float:
        """
        Calculate the CVM distance of the degree distr
        """
        #deg1 = list(self.gstats1['degree_dist'].values())
        #deg2 = list(self.gstats2['degree_dist'].values())
        dist1 = self.gstats1['degree_dist']
        dist2 = self.gstats2['degree_dist']
        union = set(self.gstats1['degree_dist']) | set(self.gstats2['degree_dist'])
        for key in union:
            dist1[key] = dist1.get(key, 0)
            dist2[key] = dist2.get(key, 0)
        deg1 = list(dist1.values())
        deg2 = list(dist2.values())

        div = scipy.stats.entropy(deg1, deg2) = scipy.stats.entropy(deg2, deg1)
        self.stats['kl_div'] = div

        return np.round(div, 3)

    #def kl_divergence(self) -> float:
    #    """
    #    Calculate the CVM distance of the degree distr
    #    """
    #    #deg1 = list(self.gstats1['degree_dist'].values())
    #    #deg2 = list(self.gstats2['degree_dist'].values())
    #    dist1 = self.gstats1['degree_dist']
    #    dist2 = self.gstats2['degree_dist']
    #    union = set(self.gstats1['degree_dist']) | set(self.gstats2['degree_dist'])
    #    for key in union:
    #        dist1[key] = dist1.get(key, 0)
    #        dist2[key] = dist2.get(key, 0)
    #    deg1 = list(dist1.values())
    #    deg2 = list(dist2.values())

    #    dist = kl_div(deg1, deg2)
    #    self.stats['kl_div'] = dist

    #    return np.round(dist, 3)
