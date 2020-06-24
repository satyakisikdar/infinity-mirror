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
from src.Tree import LightTreeNode
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
                    run_id = int(filename.split('.')[0].split('_')[-1])
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
                    try:
                        node.stats_seq
                    except AttributeError:
                        if type(node) is LightTreeNode:
                            node_graph_stats = GraphStats(run_id=run_id, graph=node.graph)
                            comparator = GraphPairCompare(GraphStats(graph=root.graph, run_id=run_id), \
                                                          GraphStats(graph=root.graph, run_id=run_id))
                            stats = {}
                            stats['lambda_dist'] = comparator.lambda_dist()
                            stats['node_diff'] = comparator.node_diff()
                            stats['edge_diff'] = comparator.edge_diff()
                            stats['pgd_pearson'] = comparator.pgd_pearson()
                            stats['pgd_spearman'] = comparator.pgd_spearman()
                            stats['deltacon0'] = comparator.deltacon0()
                            stats['degree_cvm'] = comparator.cvm_degree()
                            stats['pagerank_cvm'] = comparator.cvm_pagerank()
                            node = TreeNode(name=node.name, graph=node.graph, stats=stats, stats_seq={}, parent=node.parent, children=node.children)
                        elif type(node) is TreeNode:
                            node = TreeNode(name=node.name, graph=node.graph, stats=node.stats, stats_seq={}, parent=node.parent, children=node.children)
                        else:
                            print(f'node has unknown type: {type(node)}')
                            exit()
                    if node.stats_seq is None or node.stats_seq == {}:
                        node.stats_seq = {}
                        while len(node.children) > 0:
                            child = node.children[0]
                            try:
                                child.stats_seq = {}
                            except AttributeError:
                                child = TreeNode(name=child.name, graph=child.graph, stats=child.stats, stats_seq={}, parent=child.parent, children=child.children)
                            comparator = GraphPairCompare(GraphStats(graph=node.graph, run_id=run_id), \
                                                          GraphStats(graph=child.graph, run_id=run_id))
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
dataset = 'flights'
models = ['BUGGE']

main(base_path, dataset, models)
