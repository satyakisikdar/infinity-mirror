"""
Container for different graph stats
"""
from collections import Counter, deque
from typing import Dict, Tuple, List

import editdistance as ed
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import seaborn as sns

from src.Graph import CustomGraph
from src.utils import ColorPrint as CP

sns.set()
sns.set_style("darkgrid")


class GraphStats:
    """
    GraphStats has methods for finding different statistics for a NetworkX graph
    """
    def __init__(self, graph: CustomGraph):
        self.graph: CustomGraph = graph
        self.stats: Dict[str, float] = {'n': graph.order(), 'm': graph.size()}

    def __str__(self) -> str:
        st = f'"{self.graph.name}" stats:'
        for key, val in self.stats.items():
            if isinstance(val, float):
                val = round(val, 3)
            elif isinstance(val, dict):
                val = list(val.items())[: 3]  # print the first 3 key value pairs
            elif 'numpy' in str(type(val)):
                val = val[: 3]
            st += f'"{key}": {val} '
        return st

    def __getitem__(self, item):
        """
        Allows square bracket indexing for stats - allow for some fuzzy matching
        """
        if item in self.stats:  # the stat has already been calculated
            return self.stats[item]

        # try some fuzzy matching to figure out the function to call based on the item
        object_methods = [method_name for method_name in dir(self)
                          if callable(getattr(self, method_name)) and not method_name.startswith('_')]

        best_match_func = ''
        best_match_score = float('inf')

        for method in object_methods:
            dist = ed.eval(method, item)
            if dist == 0:
                best_match_score = dist
                best_match_func = method
                break

            if dist < best_match_score:
                best_match_score = dist
                best_match_func = method

        assert best_match_func != '', 'edit distance did not work'
        item = best_match_func
        if best_match_score != 0:
            CP.print_orange(f'Best matching function found for "{item}": "{best_match_func}()", edit distance: {best_match_score}')

        if best_match_func not in self.stats:
            best_match_func = getattr(self, best_match_func)  # translates best_match_fun from string to a function object
            best_match_func()  # call the best match function

        assert item in self.stats, f'stat: {item} is not updated after function call'
        return self.stats[item]

    def plot(self, y, ax=None, kind='line', x=None, title='', xlabel='', ylabel='') -> None:
        if isinstance(y, dict):
            lists = sorted(y.items())
            x, y = zip(*lists)
        else:  # if isinstance(x, list) or isinstance(x, np.array):
            x = list(range(len(y)))

        if kind == 'line':
            # plt.plot(x, y, marker='o', linestyle='--')
            sns.lineplot(x, y, marker='o', dashes='--', ax=ax)  # , dashes=True)
        if kind == 'scatter':
            # plt.scatter(x, y, marker='o')
            sns.scatterplot(x, y, alpha=0.75, ax=ax)
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)

        return

    def adj_eigenvalues(self):
        """
        Returns the eigenvalues of the Adjacency matrix
        :return:
        """
        CP.print_blue('Calculating eigenvalues of Adjacency Matrix')
        
        adj_eigenvalues = nx.adjacency_spectrum(self.graph)
        self.stats['adj_eigenvalues'] = adj_eigenvalues

        return adj_eigenvalues

    def assortativity(self) -> float:
        """
        Returns the assortativity of the network
        :return:
        """
        CP.print_blue('Calculating Degree Assortativity')

        assortativity = nx.degree_assortativity_coefficient(self.graph)
        self.stats['assortativity'] = assortativity

        return assortativity

    def _calculate_all_stats(self):
        """
        Calculate all stats
        """
        CP.print_orange('Calculating all stats')

        object_methods = [method_name for method_name in dir(self)
                          if callable(getattr(self, method_name)) and not method_name.startswith('_')]

        for method in object_methods:
            method = getattr(self, method)
            try:
                method()
            except NotImplementedError as e:
                pass

    def closeness_centrality(self) -> Dict[int, float]:
        """
        Closeness centrality
        """
        CP.print_blue('Calculating Closeness Centrality')

        closeness = nx.closeness_centrality(self.graph)
        self.stats['closeness_centrality'] = closeness

        return closeness

    def clustering_coefficients_by_degree(self) -> Dict[int, float]:
        """
        Returns the average clustering coefficient by degree
        :return:
        """
        CP.print_blue('Calculating Clustering Coefficients and CC by degree')

        clustering_coeffs = nx.clustering(self.graph)
        self.stats['clustering_coeffs'] = clustering_coeffs

        clustering_by_degree = Counter()  # average clustering per degree
        degree_counts = Counter()  # keeps track of #nodes with degree k

        # get the sums
        for node, cc in clustering_coeffs.items():
            deg = self.graph.degree[node]
            degree_counts[deg] += 1
            clustering_by_degree[deg] += cc

        # average it out - TODO: double check
        # for node, cc in clustering_coeffs.items():
        #     deg = self.graph.degree[node]
        #     clustering_by_degree[deg] /= degree_counts[deg]  # calculate the mean

        self.stats['clustering_coefficients_by_degree'] = clustering_by_degree
        return clustering_by_degree

    def component_size_distribution(self) -> List[Tuple[int, float]]:
        """
        Returns the distribution of component sizes and fraction of nodes in each component, largest first
        :return:
        """
        CP.print_blue('Calculating Component Size Distribution')

        component_size_ratio_list = [(len(c), len(c) / self.graph.order()) for c in
                                     sorted(nx.connected_components(self.graph),
                                            key=len, reverse=True)]
        self.stats['component_size_distribution'] = component_size_ratio_list

        return component_size_ratio_list

    def degree_centrality(self) -> Dict[int, float]:
        """
        Degree centrality
        """
        CP.print_blue('Calculating Degree Centrality')

        degree_centrality = nx.degree_centrality(self.graph)
        self.stats['degree_centrality'] = degree_centrality

        return degree_centrality

    def degree_dist(self, normalized=True) -> Dict[int, float]:
        """
        Returns the degrees counter - keys: degrees, values: #nodes with that degree
        :return:
        """
        CP.print_blue('Calculating Degree Distribution')

        degree_seq = sorted(deg for _, deg in self.graph.degree())
        self.stats['degree_seq'] = degree_seq

        degree_counts = Counter(degree_seq)

        if normalized:
            for deg, count in degree_counts.items():
                degree_counts[deg] /= self.graph.order()

        self.stats['degree_dist'] = dict(degree_counts)
        return dict(degree_counts)

    def diameter(self) -> float:
        CP.print_blue('Calculating Diameter')

        diam = nx.diameter(self.graph)
        self.stats['diameter'] = diam

        return diam

    def effective_diameter(self) -> None:
        """
        Returns the 90% effective diameter of a graph
        :return:
        """
        raise NotImplementedError()

    def k_hop_reach(self) -> np.array:
        """
        Returns the average number of nodes reachable from any node in k-hops
        Two levels of aggregation:
            1. _k_hop_reachability gives the absolute count of nodes reachable within a k-hops from a node
            2. overall_k_hop_dict aggregates the sum of all absolute counts for all nodes
        Normalizing factor: n ** 2 (once for each step)
        Then convert to a cumulative distribution
        :return:
        """
        CP.print_blue('Calculating hop-plot')

        overall_k_hop_dict = Counter()

        for node in self.graph.nodes():
            k_hop_dict = self._k_hop_reachability_counter(node)
            overall_k_hop_dict += Counter(k_hop_dict)

        k_hop_vec = np.array([v for k, v in sorted(overall_k_hop_dict.items(), key=lambda x: x[0])])
        k_hop_vec = k_hop_vec / (self.graph.order() ** 2)

        self.stats['k_hop_reach'] = np.cumsum(k_hop_vec)

        return self.stats['k_hop_reach']

    def _k_hop_reachability_counter(self, node) -> Dict[int, float]:
        """
        computes fraction of nodes reachable from the given node in k hops
        :param node: node to compute the k_hop_reach vector
        :return:
        """
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

    def laplacian_eigenvalues(self) -> np.array:
        """
        Returns eigenvalues of the Laplacian
        :return:
        """
        CP.print_blue('Calculating Laplacian Eigenvalues')

        laplacian_eigs = nx.laplacian_spectrum(self.graph)
        self.stats['laplacian_eigenvalues'] = laplacian_eigs

        return laplacian_eigs

    def pagerank(self) -> Dict[int, float]:
        """
        PageRank centrality
        """
        CP.print_blue('Calculating PageRank')

        pagerank = nx.pagerank_scipy(self.graph)
        self.stats['pagerank'] = pagerank

        return pagerank

    def pgd_graphlet_counts(self) -> None:
        """
        Return the dictionary of graphlets and their counts - based on Neville's PGD
        :return:
        """
        CP.print_blue('Calculating the graphlet counts by PGD')
        raise NotImplementedError('PGD does not work yet')
