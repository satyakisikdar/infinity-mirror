import os
import sys; sys.path.append('./..')

from src.utils import load_pickle
from src.Tree import TreeNode

in_path = '/data/infinity-mirror/pickles'
out_path = '/home/danielgonzalez/repos/infinity-mirror/output/pickles'
for subdir, dirs, files in os.walk(in_path):
    last = subdir.split('/')[-1]
    if last == 'pickles':
        last = ''
    for dir in dirs:
        try:
            os.mkdir(os.path.join(f'{out_path}/{last}', dir))
        except OSError:
            print(os.path.join(out_path, dir))
