import functools
import json
import os
import pickle
import re
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Union, Any, Tuple, List

import gzip
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.stats.api as sm

sns.set(); sns.set_style('darkgrid')

def timer(func):
    @functools.wraps(func)
    def wrapper_timer(*args, **kwargs):
        ColorPrint.print_bold(f'Start: {datetime.now().ctime()}')
        tic = time.perf_counter()
        value = func(*args, **kwargs)
        toc = time.perf_counter()
        elapsed_time = toc - tic
        ColorPrint.print_bold(f'End: {datetime.now().ctime()}')
        ColorPrint.print_bold(f'Elapsed time: {elapsed_time:0.4f} seconds')
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


def get_graph_from_prob_matrix(p_mat: np.array, thresh: float = None) -> nx.Graph:
    """
    Generates a NetworkX graph from probability matrix
    :param p_mat: matrix of edge probabilities
    :return:
    """
    n = p_mat.shape[0]  # number of rows / nodes

    if thresh is not None:
        rand_mat = np.ones((n, n)) * thresh
    else:
        rand_mat = np.random.rand(n, n)

    sampled_mat = rand_mat <= p_mat
    # sampled_mat = sampled_mat * sampled_mat.T  # to make sure it is symmetric

    sampled_mat = sampled_mat.astype(int)
    np.fill_diagonal(sampled_mat, 0)  # zero out the diagonals
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
    return


def print_float(x: float) -> float:
    """
    Prints a floating point rounded to 3 decimal places
    :param x:
    :return:
    """
    return round(x, 3)


#todoc: add some documentation to these new functions
def save_pickle(obj: Any, path: Union[Path, str]) -> Any:
    if not isinstance(path, Path):
        path = Path(path)
    ensure_dir(path.parents[0])  # ensures the parent directories exist
    return pickle.dump(obj, open(path, 'wb'), protocol=-1)  # use the highest possible protocol


# create a handler for writing sets to json (serializable)
def set_default(obj):
    if isinstance(obj, set):
        return list(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError


# write data structure to zipped json (filename should probably have a .json.gz extension)
def save_zipped_json(data: Any, filename: Union[str, Path]) -> None:
    if not isinstance(filename, Path):
        filename = Path(filename)
    ensure_dir(filename.parents[0])  # ensures the parent directories exist
    with gzip.GzipFile(filename, 'w') as fout:
        fout.write(json.dumps(data, default=set_default, indent=4).encode('utf-8'))
    return


def load_pickle(path: Union[Path, str]) -> Any:
    """
    Loads a pickle from the path
    :param path:
    :return:
    """
    assert check_file_exists(path), f'"{path}" does not exist'
    return pickle.load(open(path, 'rb'))


def load_zipped_json(filename: Union[str, Path], keys_to_int: bool = False, debug: bool = False) -> Any:
    if debug:
        ColorPrint.print_blue(f'Loading {filename!r}')
    with gzip.open(filename, 'rb') as f:
        text = f.read()
        temp = text.decode('utf-8')
        d = json.loads(temp)

    # json sadness - convert all the keys to integer, if such a thing is possible
    if isinstance(d, dict) and keys_to_int:
        d = {int(k): v for k, v in d.items()}

    return d


def load_imt_trial(input_path, dataset, model) -> (pd.DataFrame, int):
    """
    Loads graph list files and yields them to the caller one file at a time. This function loads
    each file that matches the imt_filename_pattern regex in the input directory and attempts to yield it.
    :param
        input_path: str or os.path object
        dataset:    str
        model:      str
    :return: Tuple(pd.Datafrome, int)
    """
    full_path = os.path.join(input_path, dataset, model)
    imt_filename_pattern = re.compile('list_(\d+)_(\d+).pkl.gz')
    input_filenames = [f for f in os.listdir(full_path)
                       if os.path.isfile(os.path.join(full_path, f)) and re.match(imt_filename_pattern, f)]

    for filename in input_filenames:
        imt_dataframe = load_pickle(os.path.join(full_path, filename))
        generations, trial_id = map(int, imt_filename_pattern.fullmatch(filename).groups())
        yield imt_dataframe, trial_id


def ensure_dir(path: Union[str, Path], recursive: bool=False, exist_ok: bool=True) -> None:
    path = Path(path)
    if not path.exists():
        ColorPrint.print_blue(f'Creating dir: {path!r}')
        path.mkdir(parents=recursive, exist_ok=exist_ok)
        # os.makedirs(path, exist_ok=True)
    return


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

# todo: throw these things away and consolidate 
def verify_dir(path) -> None:
    """
    Given a path, verify_dir will check if the directory exists and if not, it will create the directory.
    :param path:
    :return: None
    """
    p = Path(path)
    return os.path.exists(path)


def verify_file(path) -> bool:
    """
    Given a filepath, verify_file will return true or false depending on the existence of the file.
    :param path:
    :return: bool
    """
    return os.path.exists(path)


def get_imt_output_directory() -> os.path:
    """
    This should look in a users' home directory for a file that contains a path to that user's data directory for
    the IMT graph files.
    :param: None
    :return: data_dir: os.path
    """
    home_directory = os.environ['HOME']
    infinity_mirror_directory_file = os.path.join(home_directory, 'imt_dirs.csv').replace('\\', '/')
    path_df = pd.read_csv(infinity_mirror_directory_file)
    return path_df['output'].values[0]


def get_imt_input_directory() -> os.path:
    """
    This should look in a users' home directory for a file that contains a path to that user's data directory for
    the IMT graph files.
    :param: None
    :return: data_dir: os.path
    """
    home_directory = os.environ['HOME']
    infinity_mirror_directory_file = os.path.join(home_directory, 'imt_dirs.csv').replace('\\', '/')
    path_df = pd.read_csv(infinity_mirror_directory_file)
    return path_df['input'].values[0]


def walker():
    base_path = get_imt_input_directory()
    buckets, datasets, models, trials, filenames = [], [], [], [], []

    for subdir, dirs, files in os.walk(base_path):
        if 'bucket3' not in subdir:
            for filename in files:
                subdir_list = subdir.split('/')
                bucket = subdir_list[-3]
                dataset = subdir_list[-2]
                model = subdir_list[-1]
                trial = int(filename.split('_')[-1].strip('.pkl.gz'))

                buckets.append(bucket)
                datasets.append(dataset)
                models.append(model)
                trials.append(trial)
                filenames.append(filename)

    return buckets, datasets, models, trials, filenames

def walker_texas_ranger(fixed_stat=''):
    base_path = os.join(get_imt_output_directory(), 'graph_stats')
    datasets, models, stats, trials, iterations, filenames = [], [], [], [], []

    for subdir, dirs, files in os.walk(base_path):
        if fixed_stat in subdir:
            for filename in files:
                subdir_list = subdir.split('/')
                dataset = subdir_list[-3]
                model = subdir_list[-2]
                stat = subdir_list[-1]
                trial = int(filename.split('_')[-2])
                iteration = int(filename.split('_')[-1].strip('.json.gz'))

                datasets.append(dataset)
                models.append(model)
                stats.append(stat)
                trials.append(trial)
                iterations.append(iteration)
                filenames.append(filename)

    return datasets, models, stats, trials, iterations, filenames
