import itertools
from collections import defaultdict
from os import listdir
from os.path import join, isfile

import pandas as pd
import numpy as np
import seaborn as sns
from scipy.spatial import distance
from sklearn.neighbors import KernelDensity

if __name__ == '__main__':

    output_directory = "/data/infinity-mirror/stats/"

    input_filenames = []
    datasets = ['eucore']
    models = ['BTER']

    for dataset in datasets:
        for model in models:
            input_directory = f"/data/infinity-mirror/{dataset}/{model}/pagerank/"
            input_filenames += [input_directory + f for f in listdir(input_directory) if
                                isfile(join(input_directory, f))]

            graph_dists = defaultdict(defaultdict)

            # load all of the graphs into memory
            for filename in input_filenames:
                # print(filename)
                # parse filename for generation id
                chain_id = int(filename.split("_")[2])
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
                    current_hist = np.histogram(graph_dists[chain_id][gen_id], range=(0, abs_upper_bound), bins=100)[0] + 0.00001
                    abs_js_distance = distance.jensenshannon(original_hist, current_hist, base=2.0)

                    seq_upper_bound = max(graph_dists[chain_id][gen_id-1].max(), graph_dists[chain_id][gen_id].max())

                    pred_hist = np.histogram(graph_dists[1][0], range=(0, seq_upper_bound), bins=100)[0] + 0.00001
                    current_hist = np.histogram(graph_dists[chain_id][gen_id], range=(0, seq_upper_bound), bins=100)[0] + 0.00001
                    seq_js_distance = distance.jensenshannon(pred_hist, current_hist, base=2.0)

                    results[chain_id][gen_id] = {'abs':abs_js_distance, 'seq': seq_js_distance}

            results_df = pd.DataFrame.from_dict({(i,j): results[i][j]
                       for i in results.keys()
                       for j in results[i].keys()},
                   orient='index')

            results_df.to_csv(output_directory+f'pagerank_{dataset}_{model}.csv')

