from glob import glob
import os
from src.utils import load_pickle
import pickle
# from  import
from sys import argv
import shutil

base_path = '/data/infinity-mirror/'
dest_path = '/data/infinity-mirror/cleaned/'
datasets = ['eucore', 'flights', 'clique-ring-500-4', 'tree', 'chess']

# for dataset in datasets:
dataset = argv[1]
assert dataset in datasets

dataset_path = os.path.join(base_path, dataset)
models = set(model for model in os.listdir(dataset_path)
             if not model.startswith('_'))

for model in models:
    model_path = os.path.join(dataset_path, model)
    new_model_path = os.path.join(dest_path, dataset, model)
    if not os.path.isdir(new_model_path):
        try:
            os.mkdir(new_model_path)
        except OSError:
            print(f'ERROR: could not make directory {model_path} for some reason')

    all_pickle_files = set(file for file in os.listdir(model_path)
                           if file.endswith('.pkl.gz'))

    non_rob_pickle_files = set(file for file in all_pickle_files
                               if 'rob' not in file)

    print(f'{dataset}, {model}, {len(all_pickle_files)}, {len(non_rob_pickle_files)}')

    for root_file in non_rob_pickle_files:
        parts = root_file.split('_')
        trial_id = parts[parts.index('20') + 1]
        if trial_id.endswith('.pkl.gz'):  # if trial id is not correct for list files
            trial_id = trial_id[: trial_id.find('.pkl.gz')]

        root_path = os.path.join(model_path, root_file)
        root_dest_path = os.path.join(dest_path, dataset, model, f'list_20_{trial_id}.pkl.gz')

        if os.path.isfile(root_dest_path):
            continue
        print(f'src: {root_path} dest: {root_dest_path}')

        root = load_pickle(root_path)
        if isinstance(root, list):  # it's in the right format
            shutil.copyfile(root_path, root_dest_path)
        else:
            graphs = [root.graph]
            graphs.extend(tnode.graph for tnode in root.descendants)
            pickle.dump(graphs, open(root_dest_path, 'wb'))
    print()
print()
