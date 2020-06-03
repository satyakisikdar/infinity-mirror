import itertools
from collections import defaultdict
from os import listdir
from os.path import join, isfile

import pandas as pd
import numpy as np
import seaborn as sns
from scipy.spatial import distance
from sklearn.neighbors import KernelDensity
import scipy.stats as st

if __name__ == '__main__':

    output_directory = "/data/infinity-mirror/stats/pagerank/"

    input_filenames = []
    datasets = ['eucore']
    models = ['GCN_AE', 'Linear_AE']

    for dataset in datasets:
        for model in models:
            input_directory = f"/data/infinity-mirror/{dataset}/{model}/pagerank/"
            input_filenames = [input_directory + f for f in listdir(input_directory) if
                                isfile(join(input_directory, f))]

            graph_dists = defaultdict(defaultdict)

            # load all of the graphs into memory
            for filename in input_filenames:
                # print(filename)
                # parse filename for generation id
                if model in ['GCN_AE', 'GCN_VAE', 'Linear_AE', 'Linear_VAE', 'Deep_GCN_AE', 'Deep_GCN_VAE']:
                    chain_id = int(filename.split("_")[3].strip(".pkl.gz"))
                else:
                    chain_id = int(filename.split("_")[2].strip(".pkl.gz"))
                gen_id = int(filename.split("_")[-1].strip(".csv"))
                # print(gen_id, chain_id)
                file = pd.read_csv(filename, sep="\t")
                graph_dists[chain_id][gen_id] = file.pagerank.values

            # print(original_hist)\
            original_data = graph_dists[1][0]
            org_max = original_data.max()

            results = defaultdict(defaultdict)

            for chain_id in [x for x in graph_dists.keys() if x != 1]:
                for gen_id in [x for x in graph_dists[chain_id].keys() if x != 0]:
                    abs_upper_bound = max(org_max, graph_dists[chain_id][gen_id].max())

                    original_hist = np.histogram(graph_dists[1][0], range=(0, abs_upper_bound), bins=100)[0] + 0.00001
                    current_hist = np.histogram(graph_dists[chain_id][gen_id], range=(0, abs_upper_bound), bins=100)[
                                       0] + 0.00001
                    abs_js_distance = distance.jensenshannon(original_hist, current_hist, base=2.0)

                    seq_upper_bound = max(graph_dists[chain_id][gen_id - 1].max(), graph_dists[chain_id][gen_id].max())

                    pred_hist = np.histogram(graph_dists[1][0], range=(0, seq_upper_bound), bins=100)[0] + 0.00001
                    current_hist = np.histogram(graph_dists[chain_id][gen_id], range=(0, seq_upper_bound), bins=100)[
                                       0] + 0.00001
                    seq_js_distance = distance.jensenshannon(pred_hist, current_hist, base=2.0)

                    results[chain_id][gen_id] = {'abs': abs_js_distance, 'seq': seq_js_distance}

            results_df = pd.DataFrame.from_dict({(model, i, j): results[i][j]
                                                 for i in results.keys()
                                                 for j in results[i].keys()},
                                                orient='index')

            results_df.to_csv(output_directory + f'pagerank_{dataset}_{model}.csv')
            results_df = pd.read_csv(output_directory + f'pagerank_{dataset}_{model}.csv',
                                     names=['model', 'chain', 'gen', 'abs', 'seq'], header=0)
            results_df.to_csv(output_directory + f'pagerank_{dataset}_{model}.csv')


            def abs95u(a):
                return st.t.interval(0.95, len(a) - 1, loc=np.mean(a), scale=st.sem(a))[1]


            def abs95d(a):
                return st.t.interval(0.95, len(a) - 1, loc=np.mean(a), scale=st.sem(a))[0]


            def seq95u(a):
                return st.t.interval(0.95, len(a) - 1, loc=np.mean(a), scale=st.sem(a))[1]


            def seq95d(a):
                return st.t.interval(0.95, len(a) - 1, loc=np.mean(a), scale=st.sem(a))[0]


            def seq_mean(a):
                return np.mean(a)


            def abs_mean(a):
                return np.mean(a)


            results_df = results_df.groupby(['model', 'gen']).agg(
                {'abs': [abs_mean, abs95d, abs95u], 'seq': [seq_mean, seq95d, seq95u]})
            print(results_df.info())
            results_df.columns = results_df.columns.droplevel(0)
            results_df.to_csv(output_directory + f'pagerank_{dataset}_{model}.csv', sep='\t', index=False, na_rep='nan')

            # print(results_df.head())
# model	gen	abs_mean	abs95d	abs95u	seq_mean	seq95d	seq95u?
