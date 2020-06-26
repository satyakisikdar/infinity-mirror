"""
Reads graph stats from JSONs and calculates the distances
one CSV for each dataset, model pair

flights_lamba_dist.csv

dataset     model      trial    iteration       lambda_dist
flights     CNRG       1        0               0
...
flights     BTER       1        0              0
"""

from os.path import join
from typing import Dict, Any, Tuple, List, Union

import numpy as np
from numpy import linalg as la
from scipy.spatial import distance
from scipy.stats import entropy

from src.utils import get_imt_output_directory, load_zipped_json, verify_file


class GraphDistance:
    implemented_metrics = {'pagerank_js': 'pagerank', 'degree_js': 'degree_dist', 'pgd_distance': 'pgd_graphlet_counts', 'netlsd_distance': 'netlsd',
                           'lambda_distance': 'laplacian_eigenvalues', 'portrait_divergence': 'portrait'}

    def __init__(self, dataset: str, trial: int, model: str, metrics: List[str], iteration: Union[None, int] = None):

        self.trial = trial
        self.dataset = dataset
        self.model = model
        self.iteration = iteration
        assert all(metric in GraphDistance.implemented_metrics.keys() for metric in
                   metrics), f'Invalid metric(s) in: {metrics}, choose {GraphDistance.implemented_metrics.keys()}'
        self.stats: Dict[str, Any] = {'dataset': dataset, 'model': model, 'iteration': iteration, 'trial': trial}
        self.root = None
        self.root_metric: Union[None, str] = None
        self.total_iterations: Union[None, int] = None
        return

    def set_iteration(self, iteration: int = 0) -> None:
        self.iteration = iteration

    def set_root_object(self, metric) -> Any:
        # initialize the root object
        imt_output_directory = get_imt_output_directory()
        if not self.root or not self.root_metric == metric:
            self.root = load_zipped_json(filename=join(imt_output_directory, 'graph_stats', self.dataset, self.model,
                                                       metric, f'gs_{self.trial}_0.json.gz'), keys_to_int=True)
            # look for the last iterable file for this dataset and model combination
            for iteration in reversed(range(21)):
                filename = join(imt_output_directory, 'graph_stats', self.dataset, self.model,
                                metric, f'gs_{self.trial}_{iteration}.json.gz')
                if verify_file(filename):
                    self.total_iterations = iteration
                    self.root_metric = metric
                    break

    def get_pair_of_zipped_objects(self, metric: str) -> Tuple:
        """
        Returns a pair of objects lists/dicts - first for the root, and the second for the particular iteration
        :param metric:
        :return:
        """
        imt_output_directory = get_imt_output_directory()

        self.set_root_object(metric=metric)

        obj_iter = load_zipped_json(filename=join(imt_output_directory, 'graph_stats', self.dataset, self.model,
                                                  metric, f'gs_{self.trial}_{self.iteration}.json.gz'),
                                    keys_to_int=True)

        return self.root, obj_iter

    def lambda_distance(self, p=2) -> float:
        """
        Compare the euclidean distance between the top-k eigenvalues of the Laplacian
        :param k:
        :param p:
        :return:
        """
        lambda_seq_root, lambda_seq_iter = self.get_pair_of_zipped_objects(metric='laplacian_eigenvalues')

        k = min(len(lambda_seq_root), len(lambda_seq_iter))  # k is the smaller of the two lengths

        lambda_seq_root = np.array(lambda_seq_root[: k])  # taking the first k
        lambda_seq_iter = np.array(lambda_seq_iter[: k])

        dist = la.norm(lambda_seq_root - lambda_seq_iter, ord=p) / k
        self.stats['lambda_dist'] = dist

        return dist

    def pagerank_js(self) -> float:
        """
        Calculate the js distance of the pagerank
        """
        pagerank_dist_1, pagerank_dist_2 = map(lambda thing: list(thing.values()),
                                               self.get_pair_of_zipped_objects(metric='pagerank'))

        hist_upperbound = max(max(pagerank_dist_1), max(pagerank_dist_2))

        g1_hist = np.histogram(pagerank_dist_1, range=(0, hist_upperbound), bins=100)[0] + 0.00001
        g2_hist = np.histogram(pagerank_dist_2, range=(0, hist_upperbound), bins=100)[0] + 0.00001

        js_distance = distance.jensenshannon(g1_hist, g2_hist, base=2.0)
        self.stats['pagerank_js'] = js_distance

        return js_distance

    def degree_js(self) -> float:
        """
        Calculate the Jensen-Shannon distance of the degree distributions
        :return:
        """
        degree_dist_1, degree_dist_2 = self.get_pair_of_zipped_objects(metric='degree_dist')
        union = set(degree_dist_1.keys()) | set(degree_dist_2.keys())

        for key in union:
            degree_dist_1[key] = degree_dist_1.get(key, 0)
            degree_dist_2[key] = degree_dist_2.get(key, 0)

        deg1 = np.asarray(list(degree_dist_1.values())) + 0.00001
        deg2 = np.asarray(list(degree_dist_2.values())) + 0.00001

        degree_js = distance.jensenshannon(deg1, deg2, base=2.0)
        self.stats['degree_js'] = degree_js

        return degree_js

    def portrait_divergence(self) -> float:
        """
        :return:
        """
        b_matrix_1, b_matrix_2 = map(np.array, self.get_pair_of_zipped_objects(metric='b_matrix'))
        portrait_divergence = _calculate_portrait_divergence(b_matrix_1, b_matrix_2)

        return portrait_divergence

    def pgd_distance(self):  # todo
        raise NotImplementedError()

    def netlsd_distance(self):  # todo
        raise NotImplementedError()

    def compute_distances(self, metrics):
        for metric in metrics:
            self.set_root_object(metric=self.implemented_metrics[metric])
            func = getattr(self, metric)  # get the function obj corresponding to the metric
            func()  # call the function


def _pad_portraits_to_same_size(B1, B2):
    """
    Make sure that two matrices are padded with zeros and/or trimmed of
    zeros to be the same dimensions.
    """
    ns, ms = B1.shape
    nl, ml = B2.shape

    # Bmats have N columns, find last *occupied* column and trim both down:
    lastcol1 = max(np.nonzero(B1)[1])
    lastcol2 = max(np.nonzero(B2)[1])
    lastcol = max(lastcol1, lastcol2)
    B1 = B1[:, :lastcol + 1]
    B2 = B2[:, :lastcol + 1]

    BigB1 = np.zeros((max(ns, nl), lastcol + 1))
    BigB2 = np.zeros((max(ns, nl), lastcol + 1))

    BigB1[:B1.shape[0], :B1.shape[1]] = B1
    BigB2[:B2.shape[0], :B2.shape[1]] = B2

    return BigB1, BigB2


def _calculate_portrait_divergence(BG, BH):
    """Compute the network portrait divergence between graphs G and H."""

    # BG = _graph_or_portrait(G)
    # BH = _graph_or_portrait(H)
    BG, BH = _pad_portraits_to_same_size(BG, BH)

    L, K = BG.shape
    V = np.tile(np.arange(K), (L, 1))

    XG = BG * V / (BG * V).sum()
    XH = BH * V / (BH * V).sum()

    # flatten distribution matrices as arrays:
    P = XG.ravel()
    Q = XH.ravel()

    # lastly, get JSD:
    M = 0.55 * (P + Q)
    KLDpm = entropy(P, M, base=2)
    KLDqm = entropy(Q, M, base=2)
    JSDpq = 0.5 * (KLDpm + KLDqm)

    return JSDpq


if __name__ == '__main__':
    gd = GraphDistance(dataset='chess', iteration=13, trial=3, model='BTER', metrics=['lambda_distance', 'degree_js',
                                                                                      'portrait_divergence'])
