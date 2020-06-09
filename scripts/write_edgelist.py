import os
import sys
import pickle as pkl
import networkx as nx

def main():
    base_path = '/data/infinity-mirror/cleaned'
    dataset = 'eucore'
    models = ['Kronecker']

    output_path = '/data/infinity-mirror/cleaned/edgelists'

    for model in models:
        for subdir, dirs, files in os.walk(os.path.join(base_path, dataset, model)):
            for filename in files:
                print(f'{subdir} {filename}')
                output_filename = f'{dataset}_{model}_{filename}'.strip('.pkl.gz')
                trial_id = filename.split('_')[2]
                graphlist = pkl.load(open(os.path.join(subdir, filename), 'rb'))

                for graph in graphlist:
                    nx.write_edgelist(graph, os.path.join(output_path, output_filename + '.g'), data=False)
                    print(f'\twrote {output_path} {output_filename}.g')

    return

main()
