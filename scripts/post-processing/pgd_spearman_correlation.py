import itertools
from collections import defaultdict
from os import listdir
from os.path import join, isfile

import pandas as pd
import numpy as np
import seaborn as sns
from scipy.spatial import distance
from scipy.stats import spearmanr
from sklearn.neighbors import KernelDensity
import scipy.stats as st

from src.utils import ColorPrint

if __name__ == '__main__':

    output_directory = "../data/pgd/" # now going in the repo

    input_filenames = []
    datasets = ['eucore']
    models = ['BTER']

    for dataset in datasets:
        for model in models:
            input_directory = f"/data/infinity-mirror/stats/pgd/"
            input_filenames = [input_directory + f for f in listdir(input_directory) if
                                isfile(join(input_directory, f)) and f'{dataset}_{model}_pgd_full' in f]

            graph_dists = defaultdict(defaultdict)
            if len(input_filenames) != 1:
                ColorPrint.print_red(f'There is file inconsistancy for {dataset} using {model} \n')
                exit()

            filename = input_filenames[0]
            data = pd.read_csv(filename, sep="\t")
            ColorPrint.print_green(f'Loaded {filename}')
            # print(original_hist)\
            original_data = data.loc[(data.gen == 0) & (data.trial == 1)]
            original_data = original_data.drop(['model', 'gen', 'trial'], axis=1)
            original_data = original_data.to_numpy()[0]
            org_max = original_data.max()

            results = defaultdict(defaultdict)

            for chain_id in [x for x in data.trial.unique() if x != 1]:
                for gen_id in [x for x in data.loc[data.trial == chain_id].gen.unique() if x != 0]:
                    # abs_upper_bound = max(org_max, graph_dists[chain_id][gen_id].max())

                    current_data = data.loc[(data.gen == gen_id) & (data.trial == chain_id)].drop(['model', 'gen', 'trial'], axis=1).to_numpy()[0]
                    pred_data = data.loc[(data.gen == gen_id-1) & (data.trial == chain_id)].drop(['model', 'gen', 'trial'], axis=1).to_numpy()[0]

                    abs_spearman = spearmanr(original_data, current_data)[0]
                    seq_spearman = spearmanr(pred_data, current_data)[0]

                    results[chain_id][gen_id] = {'abs': abs_spearman, 'seq': seq_spearman}

            results_df = pd.DataFrame.from_dict({(model, i, j): results[i][j]
                                                 for i in results.keys()
                                                 for j in results[i].keys()},
                                                orient='index')

            results_df.to_csv(output_directory + f'pgd_{dataset}_{model}.csv')
            results_df = pd.read_csv(output_directory + f'pgd_{dataset}_{model}.csv',
                                     names=['model', 'chain', 'gen', 'abs', 'seq'], header=0)
            results_df.to_csv(output_directory + f'pgd_{dataset}_{model}.csv')


            def abs95u(a):
                a=a.values
                result = st.t.interval(0.95, len(a) - 1, loc=np.mean(a), scale=st.sem(a))[1]
                if np.isnan(result):
                    ColorPrint.print_red(f'CI failed on array {a} with type {type(a)}')
                    return a[0]
                return result


            def abs95d(a):
                a=a.values
                result = st.t.interval(0.95, len(a) - 1, loc=np.mean(a), scale=st.sem(a))[0]
                if np.isnan(result):
                    ColorPrint.print_red(f'CI failed on array {a}with type {type(a)}')
                    return a[0]
                return result


            def seq95u(a):
                a=a.values
                result = st.t.interval(0.95, len(a) - 1, loc=np.mean(a), scale=st.sem(a))[1]
                if np.isnan(result):
                    ColorPrint.print_red(f'CI failed on array {a}with type {type(a)}')
                    return a[0]
                return result


            def seq95d(a):
                a=a.values
                result = st.t.interval(0.95, len(a) - 1, loc=np.mean(a), scale=st.sem(a))[0]
                if np.isnan(result):
                    ColorPrint.print_red(f'CI failed on array {a}')
                    return a[0]
                return result


            def seq_mean(a):
                a=a.values
                return np.mean(a)


            def abs_mean(a):
                a=a.values
                return np.mean(a)


            results_df = results_df.groupby(['model', 'gen']).agg(
                {'abs': [abs_mean, abs95d, abs95u], 'seq': [seq_mean, seq95d, seq95u]})
            print(results_df.info())
            results_df.columns = results_df.columns.droplevel(0)
            results_df.to_csv(output_directory + f'pgd_{dataset}_{model}.csv', sep='\t', index=False, na_rep='nan')

            # print(results_df.head())
# model	gen	abs_mean	abs95d	abs95u	seq_mean	seq95d	seq95u?
