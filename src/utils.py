import functools
import os
import pickle
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Union, Any, Tuple, List

import matplotlib.pyplot as plt
import networkx as nx
import os
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

import numpy as np
import seaborn as sns
import statsmodels.stats.api as sm
from numpy import linalg as la
from scipy import sparse as sps
from scipy.sparse import issparse

sns.set()
sns.set_style("darkgrid")

def timer(func):
    @functools.wraps(func)
    def wrapper_timer(*args, **kwargs):
        ColorPrint.print_bold(f'Start: {datetime.now().ctime()}')
        tic = time.perf_counter()
        value = func(*args, **kwargs)
        toc = time.perf_counter()
        elapsed_time = toc - tic
        ColorPrint.print_bold(f'End: {datetime.now().ctime()}')
        ColorPrint.print_bold(f"Elapsed time: {elapsed_time:0.4f} seconds")
        return value
    return wrapper_timer


def get_blank_graph(name=None) -> nx.Graph:
    """
    Returns a blank graph with 1 node and 0 edges
    :return:
    """
    blank_graph = nx.empty_graph(n=1)
    gname = 'blank'
    if name is not None:
        name += f'_{name}'
    blank_graph.name = gname
    return blank_graph


def get_graph_from_prob_matrix(p_mat: np.array) -> nx.Graph:
    """
    Generates a NetworkX graph from probability matrix
    :param p_mat: matrix of edge probabilities
    :return:
    """
    n = p_mat.shape[0]  # number of rows / nodes
    sampled_mat = np.random.rand(n, n) <= p_mat
    sampled_mat = sampled_mat * sampled_mat.T  # to make sure it is symmetric
    np.fill_diagonal(sampled_mat, False)  # zero out the diagonals

    g = nx.from_numpy_array(sampled_mat, create_using=nx.Graph())
    return g

def mean_confidence_interval(arr, alpha=0.05) -> Tuple:
    if len(arr) == 1:
        return 0, 0
    return sm.DescrStatsW(arr).tconfint_mean(alpha=alpha)


def borda_sort(lists) -> List:
    """
    Finds the aggregate ranking from a list of individual rankings
    :param lists:
    :return:
    """
    scores = {}
    for l in lists:
        for idx, elem in enumerate(reversed(l)):
            if not elem in scores:
                scores[elem] = 0
            scores[elem] += idx
    return sorted(scores.keys(), key=lambda elem: scores[elem], reverse=True)


def check_file_exists(path: Union[Path, str]) -> bool:
    """
    Checks if file exists at path
    :param path:
    :return:
    """
    if isinstance(path, str):
        path = Path(path)
    return path.exists()


def delete_files(*files) -> None:
    """
    deletes all the files
    :param args:
    :return:
    """
    for file in files:
        if check_file_exists(file):
            os.remove(file)


def print_float(x: float) -> float:
    """
    Prints a floating point rounded to 3 decimal places
    :param x:
    :return:
    """
    return round(x, 3)


def load_pickle(path: Union[Path, str]) -> Any:
    """
    Loads a pickle from the path
    :param path:
    :return:
    """
    assert check_file_exists(path), f'"{path}" does not exist'
    return pickle.load(open(path, 'rb'))


def make_plot(y, kind='line', x=None, title='', xlabel='', ylabel='') -> None:
    if isinstance(y, dict):
        lists = sorted(y.items())
        x, y = zip(*lists)
    else: # if isinstance(x, list) or isinstance(x, np.array):
        x = list(range(len(y)))

    if kind == 'line':
        # plt.plot(x, y, marker='o', linestyle='--')
        sns.lineplot(x, y, marker='o', dashes='--') #, dashes=True)
    if kind =='scatter':
        # plt.scatter(x, y, marker='o')
        sns.scatterplot(x, y, alpha=0.75)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.show()

    return


def cvm_distance(data1, data2) -> float:
    data1, data2 = map(np.asarray, (data1, data2))
    n1 = len(data1)
    n2 = len(data2)
    data1 = np.sort(data1)
    data2 = np.sort(data2)
    data_all = np.concatenate([data1, data2])
    cdf1 = np.searchsorted(data1, data_all, side='right') / n1
    cdf2 = np.searchsorted(data2, data_all, side='right') / n2
    d = np.sum(np.absolute(cdf1 - cdf2))
    return np.round(d / len(cdf1), 3)


def _pad(A,N):
    """Pad A so A.shape is (N,N)"""
    n,_ = A.shape
    if n>=N:
        return A
    else:
        if issparse(A):
            # thrown if we try to np.concatenate sparse matrices
            side = sps.csr_matrix((n,N-n))
            bottom = sps.csr_matrix((N-n,N))
            A_pad = sps.hstack([A,side])
            A_pad = sps.vstack([A_pad,bottom])
        else:
            side = np.zeros((n,N-n))
            bottom = np.zeros((N-n,N))
            A_pad = np.concatenate([A,side],axis=1)
            A_pad = np.concatenate([A_pad,bottom])
        return A_pad


def fast_bp(A,eps=None):
    n, m = A.shape
    degs = np.array(A.sum(axis=1)).flatten()
    if eps is None:
        eps = 1 / (1 + max(degs))
    I = sps.identity(n)
    D = sps.dia_matrix((degs,[0]),shape=(n,n))
    # form inverse of S and invert (slow!)
    Sinv = I + eps**2*D - eps*A
    try:
        S = la.inv(Sinv)
    except:
        Sinv = sps.csc_matrix(Sinv)
        S = sps.linalg.inv(Sinv)
    return S


class ColorPrint:
    @staticmethod
    def print_red(message, end='\n'):
        sys.stderr.write('\x1b[1;31m' + message.strip() + '\x1b[0m' + end)

    @staticmethod
    def print_green(message, end='\n'):
        sys.stdout.write('\x1b[1;32m' + message.strip() + '\x1b[0m' + end)

    @staticmethod
    def print_orange(message, end='\n'):
        sys.stderr.write('\x1b[1;33m' + message.strip() + '\x1b[0m' + end)

    @staticmethod
    def print_blue(message, end='\n'):
        # pass
        sys.stdout.write('\x1b[1;34m' + message.strip() + '\x1b[0m' + end)

    @staticmethod
    def print_bold(message, end='\n'):
        sys.stdout.write('\x1b[1;37m' + message.strip() + '\x1b[0m' + end)

    @staticmethod
    def print_none(message, end='\n'):
        pass
        # sys.stdout.write(message + end)
