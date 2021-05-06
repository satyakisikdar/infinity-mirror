import functools
import gzip
import json
import os
import stat
import pickle
import re
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Union, Any, Tuple, List

import igraph as ig
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


def ensure_dir(path: Union[str, Path], recursive: bool=False, exist_ok: bool=True, make_public=True) -> None:
    path = Path(path)
    if not path.exists():
        ColorPrint.print_blue(f'Creating dir: {path!r}')
        path.mkdir(parents=recursive, exist_ok=exist_ok)
        # os.makedirs(path, exist_ok=True)
    if make_public:
        os.chmod(path=path, mode=stat.S_IRWXU | stat.S_IRWXG | stat.S_IRWXO)
    return


def verify_file(path) -> bool:
    """
    Given a filepath, verify_file will return true or false depending on the existence of the file.
    :param path:
    :return: bool
    """
    return os.path.exists(path)


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
    base_path = base_path.replace('input', 'output/pickles')
    #base_path = os.path.join(base_path, 'output', 'pickles')
    datasets, models, trials, filenames = [], [], [], []

    for subdir, dirs, files in os.walk(base_path):
        for filename in files:
            subdir_list = subdir.split('/')
            dataset = subdir_list[-2]
            model = subdir_list[-1]
            trial = int(filename.split('_')[-1].strip('.pkl.gz'))

            datasets.append(dataset)
            models.append(model)
            trials.append(trial)
            filenames.append(filename)

    return datasets, models, trials, filenames

def walker_prime():
    base_path = get_imt_input_directory()
    buckets, datasets, models, trials, filenames = [], [], [], [], []

    for subdir, dirs, files in os.walk(base_path):
        if 'bucket3' not in subdir and 'chess' not in subdir:
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

def walker_texas_ranger(dataset='eucore', model='BTER', stat='pagerank', unique=False):
    base_path = os.path.join(get_imt_output_directory(), 'graph_stats', dataset, model, stat)
    trials, iterations, filenames = [], [], []

    for subdir, dirs, files in os.walk(base_path):
        for filename in files:
            subdir_list = subdir.split('/')
            dataset = subdir_list[-3]
            model = subdir_list[-2]
            stat = subdir_list[-1]
            trial = int(filename.split('_')[-2])
            iteration = int(filename.split('_')[-1].strip('.json.gz'))

            trials.append(trial)
            iterations.append(iteration)
            filenames.append(filename)

    if unique:
        return list(np.unique(trials))
    return trials, iterations, filenames

def walker_michigan(dataset='eucore', model='BTER', stat='b_matrix'):
    assert dataset in ['chess', 'clique-ring-500-4', 'eucore', 'flights', 'tree']
    assert model in ['BTER', 'BUGGE', 'Chung-Lu', 'CNRG', 'Erdos-Renyi', 'GCN_AE', 'GraphRNN', 'HRG', 'Kronecker', 'Linear_AE', 'NetGAN', 'SBM']
    assert stat in ['b_matrix', 'degree_dist', 'laplacian_eigenvalues', 'netlsd', 'pagerank', 'pgd_graphlet_counts', 'average_path_length', 'average_clustering']

    base_path = f'/data/infinity-mirror/output/graph_stats/{dataset}/{model}/{stat}'

    for subdir, dirs, files in os.walk(base_path):
        for filename in files:
            filepath = os.path.join(base_path, filename)
            split = filename.split('.')[0].split('_')
            if len(filename) < 3 or filename[0] == '.':
                ColorPrint.print_red(f'CAUTION: Skipped {filename}')
                continue
            trial, gen = split[1], split[2]
            yield filepath, int(trial), int(gen)

    return


def nx_to_igraph(nx_g) -> ig.Graph:
    """
    Convert networkx graph to an equivalent igraph Graph
    attributes are stored as vertex sequences
    """
    nx_g = nx.convert_node_labels_to_integers(nx_g, label_attribute='old_label')
    old_label = nx.get_node_attributes(nx_g, 'old_label')

    weights = nx.get_edge_attributes(nx_g, name='wt')  # WEIGHTS are stored in WT
    if len(weights) == 0:
        is_weighted = False
        edge_list = list(nx_g.edges())
    else:
        is_weighted = True
        edge_list = [(u, v, w) for (u, v), w in weights.items()]

    is_directed = nx_g.is_directed()
    ig_g = ig.Graph.TupleList(edges=edge_list, directed=is_directed,
                              weights=is_weighted)

#     logging.error(f'iGraph: n={ig_g.vcount()}\tm={ig_g.ecount()}\tweighted={is_weighted}\tdirected={is_directed}')

    for v in ig_g.vs:
        v['name'] = str(old_label[v.index])  # store the original labels in the name attribute
        v['label'] = str(v['name'])

    return ig_g

def latex_mono_printer(path):
    def _header(outfile):
        outfile.write('\\pgfplotstableread{\n')
        outfile.write('model\tgen\tabs_mean\tabs95d\tabs95u\n')
        return

    def _footer(outfile, dataset, model, stat):
        outfile.write('}{\\' + dataset + stat + model + '}\n')

    dataset_map = {'chess': 'chess', 'clique-ring-500-4': 'cliquering', 'eucore': 'eucore', 'flights': 'flights', 'tree': 'tree'}
    model_map = {'BTER': 'BTER', 'BUGGE': 'BUGGE', 'Chung-Lu': 'CL', 'CNRG': 'CNRG', 'Erdos-Renyi': 'ER', 'HRG': 'HRG', 'Kronecker': 'Kron', 'NetGAN': 'NetGAN', 'SBM': 'SBM', 'GCN_AE': 'GCNAE', 'Linear_AE': 'LinearAE', 'GraphRNN': 'GraphRNN'}
    stat_map = {'degree_js': 'degree', 'lambda_dist': 'lambda', 'netlsd': 'netlsd', 'pagerank_js': 'pagerank', 'pgd_rgfd': 'pgd', 'portrait_js': 'portrait', 'avg_pl': 'apl', 'avg_clustering': 'clustering'}
    check_map = {}

    for key, value in model_map.items():
        check_map[value] = True

    filename = path.split('/')[-1].split('.')[0]

    with open(path) as infile, open(f'data_latex/{filename}.tex', 'w') as outfile:
        prev_model = ''

        for line in infile:
            line = line.strip().split('\t')

            if line[0] == 'dataset':
                _header(outfile)
                stat = stat_map[line[3]]
            else:
                dataset = dataset_map[line[0]]
                model = model_map[line[1]]
                gen = line[2]
                abs_mean = line[3]
                abs95d = line[4]
                abs95u = line[5]

                check_map[model] = False

                if prev_model != model:
                    _footer(outfile, dataset, prev_model, stat)
                    _header(outfile)

                outfile.write(f'{model}\t{gen}\t{abs_mean}\t{abs95d}\t{abs95u}\n')
                prev_model = model

        _footer(outfile, dataset, model, stat)

        for key, value in check_map.items():
            if value:
                _header(outfile)
                _footer(outfile, dataset, key, stat)
    return

def latex_bi_printer(path):
    def _header(outfile, stat1, stat2):
        outfile.write('\\pgfplotstableread{\n')
        outfile.write(f'dataset\tmodel\tgen\t{stat1}\t{stat2}\n')
        return

    def _footer(outfile, dataset, model, stat1, stat2):
        outfile.write('}{\\' + model + dataset + stat_map[(stat1, stat2)] + '}\n')

    dataset_map = {'chess': 'chess', 'clique-ring-500-4': 'cr', 'eucore': 'eucore', 'flights': 'flights', 'tree': 'tree'}
    model_map = {'BTER': 'BTER', 'BUGGE': 'BUGGE', 'Chung-Lu': 'ChungLu', 'CNRG': 'CNRG', 'Erdos-Renyi': 'ErdosRenyi', 'HRG': 'HRG', 'Kronecker': 'Kronecker', 'NetGAN': 'NetGAN', 'SBM': 'SBM', 'GCN_AE': 'GCNAE', 'Linear_AE': 'LinearAE', 'GraphRNN': 'GraphRNN'}
    stat_map = {('clu', 'pl'): 'APLCC'}
    check_map = {}

    for key, value in model_map.items():
        check_map[value] = True

    filename = path.split('/')[-1].split('.')[0]

    with open(path) as infile, open(f'data_latex/{filename}.tex', 'w') as outfile:
        prev_model = ''

        for line in infile:
            line = line.strip().split('\t')

            if line[0] == 'dataset':
                stat1 = line[3]
                stat2 = line[4]
                _header(outfile, stat1, stat2)
            else:
                dataset = dataset_map[line[0]]
                model = model_map[line[1]]
                gen = line[2]
                mean1 = line[3]
                mean2 = line[4]

                check_map[model] = False

                if prev_model != model:
                    _footer(outfile, dataset, prev_model, stat1, stat2)
                    _header(outfile, stat1, stat2)

                outfile.write(f'{dataset}\t{model}\t{gen}\t{mean1}\t{mean2}\n')
                prev_model = model

        _footer(outfile, dataset, model, stat1, stat2)

        for key, value in check_map.items():
            if value:
                _header(outfile, stat1, stat2)
                _footer(outfile, dataset, key, stat1, stat2)
    return
