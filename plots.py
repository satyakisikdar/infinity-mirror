import networkx as nx
from glob import glob
import pdb
import re
import seaborn as sns; sns.set()
import matplotlib.pyplot as plt
from matplotlib import gridspec
import pandas as pd; pd.options.display.float_format = '{:,.2f}'.format
import statsmodels.stats.api as sm
import warnings; warnings.filterwarnings("ignore", category=UserWarning)
from statistics import median_low

import sys
import os

#sys.path.extend(['./..'])  # have to add the project path manually to the Python path
#os.chdir('./..')

from src.utils import load_pickle
from src.Tree import TreeNode

def get_stats_from_root(graph, model, sel, root, cols, trial_id):
    for tnode in root.descendants:
        row = {}
        row['graph'] = graph
        row['type'] = 'absolute'

        row['orig_n'] = root.graph.order()
        row['orig_m'] = root.graph.size()
        row['orig_graph_obj'] = root.graph

        row['model'] = model
        row['sel'] = sel
        row['trial_id'] = trial_id

        row['gen_id'] = tnode.depth
        row['gen_n'] = tnode.graph.order()
        row['gen_m'] = tnode.graph.size()
        row['gen_graph_obj'] = tnode.graph

        # use the stats compared with the original seed
        stats = tnode.stats
        assert set(cols[-8: ]) == set(stats.keys()), f'tnode: {stats.keys()} doesnt have all the reqd stats'
        for key, val in stats.items():
            row[key] = val

        assert len(row.keys()) == len(cols), \
            f'Improper number of cols in row: {len(row)}: expected {len(cols)} {stats.keys()}'

        yield row
    for tnode in root.descendants:
        row = {}
        row['graph'] = graph
        row['type'] = 'sequential'

        row['orig_n'] = root.graph.order()
        row['orig_m'] = root.graph.size()
        row['orig_graph_obj'] = root.graph

        row['model'] = model
        row['sel'] = sel
        row['trial_id'] = trial_id

        row['gen_id'] = tnode.depth
        row['gen_n'] = tnode.graph.order()
        row['gen_m'] = tnode.graph.size()
        row['gen_graph_obj'] = tnode.graph

        # use the stats compared with the previous graph
        stats = tnode.stats_seq
        assert set(cols[-8: ]) == set(stats.keys()), f'tnode: {stats.keys()} doesnt have all the reqd stats'
        for key, val in stats.items():
            row[key] = val

        assert len(row.keys()) == len(cols), \
            f'Improper number of cols in row: {len(row)}: expected {len(cols)} {stats.keys()}'

        yield row

def group_plot(df, graph_name, model_name, save_path):
    graph = df.graph.unique()[0]
    metrics = ['node_diff', 'edge_diff', 'lambda_dist', 'deltacon0', 'degree_cvm']#, 'pgd_spearman']
    models = df.model.unique()
    rows = len(metrics)
    cols = len(models)

    n_d_min = min(df[df.model==model].node_diff.min() for model in models) - 1
    n_d_max = max(df[df.model==model].node_diff.max() for model in models) + 5

    e_d_min = min(df[df.model==model].edge_diff.min() for model in models) - 1
    e_d_max = max(df[df.model==model].edge_diff.max() for model in models) + 5

    l_d_min = min(df[df.model==model].lambda_dist.min() for model in models) - 0.1
    l_d_max = max(df[df.model==model].lambda_dist.max() for model in models) + 0.15

    dc0_min = min(df[df.model==model].deltacon0.min() for model in models) - 100
    dc0_max = max(df[df.model==model].deltacon0.max() for model in models) + 100

    p_sp_min = min(df[df.model==model].pgd_spearman.min() for model in models) - 0.1
    p_sp_max = max(df[df.model==model].pgd_spearman.max() for model in models) + 0.15

    d_min = min(df[df.model==model].degree_cvm.min() for model in models) - 0.1
    d_max = max(df[df.model==model].degree_cvm.max() for model in models) + 0.15

    fig, axes = plt.subplots(nrows=rows, ncols=cols, sharex=True)
    #print(rows, cols)

    for i in range(rows):
        for j in range(cols):
            ax = axes[i]
            #ax = axes[i, j]
            metric = metrics[i]
            model = models[j]
            filtered_df = df[df.model==model]

            if i == 0 and j == 0:
                legend_style = 'brief'
            else:
                legend_style = ''
            sns.lineplot(x='gen_id', y=metric, ax=ax, data=filtered_df,
                         hue='type', marker='o', ci=99, err_style='band', legend=legend_style);
            if metric == 'node_diff':
                ax.set_ylim((n_d_min, n_d_max))
            elif metric == 'edge_diff':
                ax.set_ylim((e_d_min, e_d_max))
            elif metric == 'lambda_dist':
                ax.set_ylim((l_d_min, l_d_max))
            elif metric == 'deltacon0':
                ax.set_ylim((dc0_min, dc0_max))
            elif metric == 'pgd_spearman':
                ax.set_ylim((p_sp_min, p_sp_max))
            elif metric == 'degree_cvm':
                ax.set_ylim((d_min, d_max))

            if j == 0:
                ax.set_ylabel(metric)
            else:
                ax.set_ylabel('')

            if i == 0:
                ax.set_title(model)
            else:
                ax.set_title('')

            if i == rows - 1:
                ax.set_xlabel('gen_id')
            else:
                ax.set_xlabel('')

    plt.suptitle(f'{graph}', y=1.03);
    plt.tight_layout()
    #plt.savefig(f'analysis/figures/{graph_name}.pdf', format='pdf', dpi=1000, bbox_inches='tight')
    plt.savefig(save_path + f'{graph_name}-{model_name}.pdf', format='pdf', dpi=1000, bbox_inches='tight')

def plot(df, graph, save_path):
    return

def main():
    # INVARIANTS
    graphs = ['eucore', 'clique-ring-500-4']
    generators = ['BTER', 'BUGGE', 'Chung-Lu', 'CNRG', \
                  'Erdos-Renyi', 'HRG', 'SBM']
    kronecker = ['Kronecker']
    autoencoders = ['Deep_GCN_AE', 'Deep_GCN_VAE', \
                    'GCN_AE', 'GCN_VAE', \
                    'Linear_AE', 'Linear_VAE']
    neural = ['NetGAN'] # add GraphRNN when it's ready

    # VARIANTS
    data_path = '/Users/akira/repos/infinity-mirror/output'
    models = neural
    graph = graphs[1]
    sel = 'fast'
    cols = ['graph', 'type', 'orig_n', 'orig_m', 'orig_graph_obj', \
            'model', 'sel', 'trial_id', \
            'gen_id', 'gen_n', 'gen_m', 'gen_graph_obj', \
            'deltacon0', 'lambda_dist', 'degree_cvm', 'pagerank_cvm', \
            'pgd_pearson', 'pgd_spearman', 'node_diff', 'edge_diff']

    plt.rcParams['figure.figsize'] = [10, 20]
    #data = {col: [] for col in cols}

    for model in models:
        data = {col: [] for col in cols}
        path = os.path.join(data_path, graph, model)
        print(f'reading: {model}... ', end='', flush=True)
        for filename in os.listdir(path):
            if filename[5:7:1] == '20':
                trial_id = filename[8:10:1]
                try:
                    trial_id = int(trial_id)
                except ValueError:
                    trial_id = int(trial_id[:-1])
                root = load_pickle(os.path.join(path, filename))
                for row in get_stats_from_root(graph=graph, \
                                               model=model, \
                                               sel=sel, \
                                               root=root, \
                                               cols=cols, \
                                               trial_id=trial_id):
                    for col, val in row.items():
                        data[col].append(val)
        print('done')
        group_plot(pd.DataFrame(data), graph, model, f'/Users/akira/figures/{graph}/')

    #df = pd.DataFrame(data)

    # FIGURE SIZE

    return

main()
