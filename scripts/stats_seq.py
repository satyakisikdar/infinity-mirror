import os
import sys; sys.path.append('./..')
import pickle
import networkx as nx # requires 2.3.0
import pandas as pd; pd.options.display.float_format = '{:,.2f}'.format
import statsmodels.stats.api as sm
import warnings; warnings.filterwarnings("ignore", category=UserWarning)
from glob import glob
from statistics import median_low

from src.Tree import TreeNode
from src.utils import load_pickle
from src.graph_stats import GraphStats
from src.graph_comparison import GraphPairCompare

#graphs = ['eucore', 'clique-ring-500-4', 'flights', 'tree', 'chess']
def main(base_path, dataset, models):
    if 'GraphRNN' in models:
        #path = os.path.join(base_path, 'GraphRNN')
        #for subdir, dirs, files in os.walk(path):
        #    if dataset == subdir.split('/')[-1].split('_')[0]:
        #        print(subdir)
        #        for filename in files:
        #            print(filename)
        models.remove('GraphRNN')
    for model in models:
        path = os.path.join(base_path, dataset, model)
        for subdir, dirs, files in os.walk(path):
            for filename in files:
                if 'seq' not in filename:
                #if 'augmented' not in filename:
                    string = subdir.split('/')[-2:]
                    file = os.path.join(subdir, filename)
                    newfile = file.split('.')[0]
                    if 'rob' in file:
                        newfile += '_seq_rob.pkl.gz'
                    else:
                        newfile += '_seq.pkl.gz'
                    print(f'starting\t{string[-2]}\t{string[-1]}\t{filename} ... ', end='', flush=True)
                    root = load_pickle(file)
                    node = root
                    if node.stats_seq is None or node.stats_seq == {}:
                        node.stats_seq = {}
                        while len(node.children) > 0:
                            child = node.children[0]
                            comparator = GraphPairCompare(GraphStats(graph=node.graph, run_id=0),\
                                                            GraphStats(graph=child.graph, run_id=0))
                            child.stats_seq = {}
                            child.stats_seq['lambda_dist'] = comparator.lambda_dist()
                            child.stats_seq['node_diff'] = comparator.node_diff()
                            child.stats_seq['edge_diff'] = comparator.edge_diff()
                            child.stats_seq['pgd_pearson'] = comparator.pgd_pearson()
                            child.stats_seq['pgd_spearman'] = comparator.pgd_spearman()
                            child.stats_seq['deltacon0'] = comparator.deltacon0()
                            child.stats_seq['degree_cvm'] = comparator.cvm_degree()
                            child.stats_seq['pagerank_cvm'] = comparator.cvm_pagerank()
                            node = child
                    with open(newfile, 'wb') as f:
                        pickle.dump(root, f)
                    print(f'\tdone')

base_path = '/data/dgonza26'
dataset = 'chess'
models = ['BTER', 'BUGGE', 'Chung-Lu', 'CNRG', 'Erdos-Renyi', 'HRG', 'SBM']

main(base_path, dataset, models)
