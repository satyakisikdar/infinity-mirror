"""
Graph Comparison Functions
"""
from math import fabs
from typing import Dict, List

import networkx as nx
import numpy as np
import scipy.stats
from numpy import linalg as la
from scipy import sparse as sps
from scipy.sparse import issparse
from scipy.spatial import distance

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
        # self.calculate()
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
        self.js_distance()

        if self.graph2.order() == 1 or 'blank' in self.graph2.name:  # empty graph
            for key in self.stats:
                self.stats[key] = float('inf')  # set the distances to infinity
        return

    def pgd_spearman(self) -> float:
        raise NotImplementedError()

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
        Compare the euclidean distance between the top-k eigenvalues of the Laplacian
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

    # todo (trenton)
    def pagerank_js(self) -> float:
        """
        Calculate the js distance of the pagerank
        """
        g1_dist = self.gstats1['pagerank']
        g2_dist = self.gstats2['pagerank']

        hist_upperbound = max(g1_dist.max(), g2_dist.max())

        g1_hist = np.histogram(g1_dist, range=(0, hist_upperbound), bins=100)[0] + 0.00001
        g2_hist = np.histogram(g2_dist, range=(0, hist_upperbound), bins=100)[0] + 0.00001

        js_distance = distance.jensenshannon(g1_hist, g2_hist, base=2.0)
        self.stats['pagerank_js'] = js_distance

        return js_distance

    def degree_js(self) -> float:
        """
        Calculate the Jensen-Shannon distance of the degree distributions
        :return:
        """
        dist1 = self.gstats1['degree_dist']
        dist2 = self.gstats2['degree_dist']
        union = set(dist1.keys()) | set(dist2.keys())

        for key in union:
            dist1(key) = dist1.get(key, 0)
            dist2(key) = dist2.get(key, 0)

        deg1 = np.asarray(list(dist1.values())) + 0.00001
        deg2 = np.asarray(list(dist2.values())) + 0.00001

        degree_js = distance.jensenshannon(deg1, deg2, base=2.0)
        self.stats['degree_js'] = degree_js

        return degree_js

    # todo portrait (trenton)
    def portrait_divergence(self) -> float:
        """

        :return:
        """
        raise NotImplementedError()

    def embedding_distance(self) -> float:
        """
        Calculate the Euclidean distance between two NetLSD embedding vectors
        :return:
        """
        vec1 = np.asarray(self.gstats1['netlsd'])
        vec2 = np.asarray(self.gstats2['netlsd'])

        L2 = np.sqrt(np.sum(np.square(vec1 - vec2)))
        self.stats['embedding_distance'] = L2

        return L2

# todo maybe get rid of this?
def js_distance(vec1: list, vec2: list):
    js_distance = distance.jensenshannon(vec1, vec2, base=2.0)

    return np.round(js_distance, 3)


def cvm_distance(data1: list, data2: list) -> float:
    data1, data2 = map(np.asarray, (data1, data2))
    n1 = len(data1)
    n2 = len(data2)
    data1 = np.sort(data1)
    data2 = np.sort(data2)
    data_all = np.concatenate([data1, data2])
    cdf1 = np.searchsorted(data1, data_all, side='right') / n1
    cdf2 = np.searchsorted(data2, data_all, side='right') / n2
    assert len(cdf1) == len(cdf2), 'CDFs should be of the same length'
    d = np.sum(np.absolute(cdf1 - cdf2)) / len(cdf1)
    return np.round(d, 3)


def ks_distance(data1, data2) -> float:
    data1, data2 = map(np.asarray, (data1, data2))
    n1 = len(data1)
    n2 = len(data2)
    data1 = np.sort(data1)
    data2 = np.sort(data2)
    data_all = np.concatenate([data1, data2])
    cdf1 = np.searchsorted(data1, data_all, side='right') / n1
    cdf2 = np.searchsorted(data2, data_all, side='right') / n2
    d = np.max(np.absolute(cdf1 - cdf2))
    return np.round(d, 3)


def _pad(A, N):
    """Pad A so A.shape is (N,N)"""
    n, _ = A.shape
    if n >= N:
        return A
    else:
        if issparse(A):
            # thrown if we try to np.concatenate sparse matrices
            side = sps.csr_matrix((n, N - n))
            bottom = sps.csr_matrix((N - n, N))
            A_pad = sps.hstack([A, side])
            A_pad = sps.vstack([A_pad, bottom])
        else:
            side = np.zeros((n, N - n))
            bottom = np.zeros((N - n, N))
            A_pad = np.concatenate([A, side], axis=1)
            A_pad = np.concatenate([A_pad, bottom])
        return A_pad


def fast_bp(A, eps=None):
    n, m = A.shape
    degs = np.array(A.sum(axis=1)).flatten()
    if eps is None:
        eps = 1 / (1 + max(degs))
    I = sps.identity(n)
    D = sps.dia_matrix((degs, [0]), shape=(n, n))
    # form inverse of S and invert (slow!)
    Sinv = I + eps ** 2 * D - eps * A
    try:
        S = la.inv(Sinv)
    except:
        Sinv = sps.csc_matrix(Sinv)
        S = sps.linalg.inv(Sinv)
    return S
