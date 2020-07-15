import csv
import os
import re
from collections import namedtuple
from os.path import join
from typing import Any, List

import networkx as nx
from tqdm import tqdm

from src.utils import ColorPrint as CP, load_pickle, check_file_exists, delete_files, save_pickle

Stats = namedtuple('Stats',
                   'name id graph score')  # stores the different stats for each graph. name: name of metric, id: graph_id
GraphStatDouble = namedtuple('GraphStatDouble',
                             'graph stats')  # graph: NetworkX object, stat: dictionary of comparison stats with the input
GraphStatTriple = namedtuple('GraphStatTriple',
                             'best worst median')  # stores the best, worst, and median graphs and their stats (GraphStat double)


class InfinityMirror:
    """
    Class for InfinityMirror
    """
    __slots__ = ('initial_graph', 'num_generations', 'num_graphs', 'model', 'initial_graph_stats', 'graphs', 'features',
                 '_metrics', 'graphs_pickle_path', 'graphs_features_path', 'trial', 'rewire', 'finish_path')

    def __init__(self, initial_graph: nx.Graph, model_obj: Any, num_generations: int,
                 num_graphs: int, trial: int, r: float, dataset: str, model: str, finish: str='', features_bool: bool=False) -> None:
        self.initial_graph: nx.Graph = initial_graph  # the initial starting point H_0
        self.num_graphs: int = num_graphs  # number of graphs per generation
        self.num_generations: int = num_generations  # number of generations
        self.trial = trial
        self.model = model_obj  # (input_graph=self.initial_graph, trial=self.trial)
        self.model.input_graph = self.initial_graph
        self.model.trial = trial  # initialize and fit the model

        # todo figure out if we will remove this line or add `iteration` and `trial` information
        #self.initial_graph_stats: GraphStats = GraphStats(graph=self.initial_graph, dataset=dataset, model=model, iteration=, trial=)
        self._metrics: List[str] = ['deltacon0', 'lambda_dist', 'pagerank_cvm', 'node_diff', 'edge_diff', 'pgd_pearson',
                                    'pgd_spearman', 'degree_cvm']  # list of metrics  ## GCD is removed
        self.rewire = int(r * 100)
        self.graphs_pickle_path: str = f'./output/pickles/{self.initial_graph.name}/{self.model.model_name}/list_{self.num_generations}_{self.trial}'
        self.graphs_features_path: str = f'./output/features/{self.initial_graph.name}/{self.model.model_name}/list_{self.num_generations}_{self.trial}'
        self.graphs: List[nx.Graph] = []  # stores the list of graphs - one per generation
        self.features: List[Any] = []  # stores the learned features used to generate the graph at the same index in self.graphs - one per generation
        self.features_bool: bool = features_bool  # decides whether features are going to be extracted or not
        self.finish_path: str = finish
        if r != 0:
            self.graphs_pickle_path += f'_{r}'
        return

    def __str__(self) -> str:
        return f'({self.trial}) model: "{self.model.model_name}"  initial graph: "{self.initial_graph.name}"  #gens: {self.num_generations}'

    def __repr__(self) -> str:
        return str(self)

    def run(self, use_pickle: bool) -> None:
        """
        New runner - uses list of graphs
        :param use_pickle:
        :return:
        """
        pickle_ext = '.pkl.gz'
        self.graphs = []

        if use_pickle:
            if check_file_exists(self.graphs_pickle_path + pickle_ext):  # the whole pickle exists
                graphs = load_pickle(self.graphs_pickle_path + pickle_ext)
                assert len(graphs) == 21, f'Expected 21 graphs, found {len(graphs)}'
                CP.print_green(f'Using completed pickle at {self.graphs_pickle_path + pickle_ext!r}. Loaded {len(graphs)} graphs')
                return
            else:
                temp_file_pattern = re.compile(f'list_(\d+)_{self.trial}_temp_(\d+).pkl.gz')
                dir_name = '/'.join(self.graphs_pickle_path.split('/')[: -1])

                input_files = [f for f in os.listdir(dir_name) if re.match(temp_file_pattern, f)]
                if len(input_files) > 0:
                    assert len(input_files) == 1, f'More than one matches found: {input_files}'

                    input_file = input_files[0]
                    total_generations, progress = map(int, temp_file_pattern.fullmatch(input_file).groups())
                    graphs = load_pickle(join(dir_name, input_file))
                    assert len(graphs) == progress + 1, f'Found {len(graphs)}, expected: {progress}'
                    CP.print_blue(f'Partial pickle found at {input_file!r} trial: {self.trial} progress: {progress}/{total_generations}')
                    self.graphs = graphs

        remaining_generations = self.num_generations - len(self.graphs)

        tqdm.write(
            f'Running Infinity Mirror on {self.initial_graph.name!r} {self.initial_graph.order(), self.initial_graph.size()} {self.model.model_name!r} {remaining_generations} generations')
        pbar = tqdm(total=remaining_generations, bar_format='{l_bar}{bar}|[{elapsed}<{remaining}]', ncols=50)

        if len(self.graphs) == 0:
            self.initial_graph.level = 0
            self.graphs = [self.initial_graph]
            self.features = [None]

        completed_trial = False
        for i in range(len(self.graphs) - 1, self.num_generations):
            if i == len(self.graphs) - 1:
                curr_graph = self.graphs[-1]  # use the last graph

            level = i + 1
            try:
                self.model.update(new_input_graph=curr_graph)  # update the model
            except Exception as e:
                print(f'Model fit failed {e}')
                break

            try:
                generated_graphs = self.model.generate(num_graphs=self.num_graphs, gen_id=level)  # generate a new set of graphs
            except Exception as e:
                print(f'Generation failed {e}')
                break

            if self.features:
                self.features.append(self.model.params)
            curr_graph = generated_graphs[0]  # we are only generating one graph
            curr_graph.name = f'{self.initial_graph.name}_{level}_{self.trial}'
            curr_graph.gen = level
            self.graphs.append(curr_graph)

            temp_pickle_path = self.graphs_pickle_path + f'_temp_{level}{pickle_ext}'
            prev_temp_pickle_path = self.graphs_pickle_path + f'_temp_{level-1}{pickle_ext}'
            temp_features_path = self.graphs_features_path + f'_temp_{level}{pickle_ext}'
            prev_temp_features_path = self.graphs_features_path + f'_temp_{level-1}{pickle_ext}'
            save_pickle(obj=self.graphs, path=temp_pickle_path)
            save_pickle(obj=self.params, path=temp_features_path)
            delete_files(prev_temp_pickle_path)
            delete_files(prev_temp_features_path)

            if level == 20:
                completed_trial = True
            pbar.update(1)
        pbar.close()

        if completed_trial:  # only delete the temp pickle if the trial finishes successfully
            delete_files(temp_pickle_path)  # delete the temp file if the loop finishes normally
            CP.print_green(f'List of {len(self.graphs)} Graphs is pickled at "{self.graphs_pickle_path + pickle_ext}"')
            save_pickle(obj=self.graphs, path=self.graphs_pickle_path + pickle_ext)
        return

    def write_timing_stats(self, time_taken) -> None:
        """
        Write timing stats into a csv
        Write model info and timing info
        :return:
        """
        fieldnames = ['trial', 'gname', 'model', 'sel', 'gens', 'time']

        stats_file = './output/timing_stats.csv'
        if not check_file_exists(stats_file):  # initialize the file with headers
            writer = csv.DictWriter(open(stats_file, 'w'), fieldnames=fieldnames)
            writer.writeheader()

        with open(stats_file, 'a') as fp:
            writer = csv.DictWriter(fp, fieldnames=fieldnames)
            writer.writerow({'trial': self.trial, 'gname': self.initial_graph.name, 'model': self.model.model_name,
                             'gens': self.num_generations, 'time': time_taken})

        return

    def write_fail_stats(self, level) -> None:
        """
        Write fail stats into a csv
        :return:
        """
        fieldnames = ['trial', 'gname', 'model', 'sel', 'gens', 'level']

        fail_file = './output/fail_stats.csv'
        if not check_file_exists(fail_file):  # initialize the file with headers
            writer = csv.DictWriter(open(fail_file, 'w'), fieldnames=fieldnames)
            writer.writeheader()

        with open(fail_file, 'a') as fp:
            writer = csv.DictWriter(fp, fieldnames=fieldnames)
            writer.writerow({'trial': self.trial, 'gname': self.initial_graph.name, 'model': self.model.model_name,
                             'gens': self.num_generations, 'level': level})

        return
