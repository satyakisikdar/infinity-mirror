import os
import sys; sys.path.append('./..')
import pickle

from src.utils import load_pickle
from src.Tree import TreeNode

data_path = '/data/infinity-mirror/pickles'
base_path = '/home/danielgonzalez/repos/infinity-mirror/output/pickles'
for subdir, dirs, files in os.walk(data_path):
    tail = subdir[30:]
    for file in files:
        if 'fast' in file:
            infile = os.path.join(data_path, tail, file)
            outfile = os.path.join(base_path, tail, file)
            print(f'start\t{infile}')
            try:
                old_root = load_pickle(infile)
                new_root = TreeNode(name=old_root.name,\
                                    stats=old_root.stats,\
                                    stats_seq={},\
                                    graph=old_root.graph)
                node = new_root
                while len(old_root.children) > 0:
                    old_root = old_root.children[0]
                    node = TreeNode(name=old_root.name,\
                                    stats=old_root.stats,\
                                    graph=old_root.graph,\
                                    parent=node)
                with open(outfile, 'wb') as f:
                    pickle.dump(new_root, f)
                print(f'done\t{outfile}')
            except ModuleNotFoundError:
                print(f'ERROR: {infile}')
