from collections import Counter
import os
import sys; sys.path.append('./../../')
import pickle
import numpy as np
import pandas as pd
import networkx as nx
import scipy.stats as st
import multiprocessing as mp
from pathlib import Path
from src.Tree import TreeNode
from src.utils import load_pickle
from src.graph_stats import GraphStats
from src.graph_comparison import GraphPairCompare

def load_df(path):
    for subdir, dirs, files in os.walk(path):
        for filename in files:
            if 'deltacon' in filename and 'csv' in filename:
                print(f'\tloading {subdir} {filename} ... ', end='', flush=True)
                df = pd.read_csv(os.path.join(subdir, filename), sep='\t'), filename
                print('done')
                yield df

def mkdir_output(path):
    if not os.path.isdir(path):
        try:
            os.mkdir(path)
        except OSError:
            print('ERROR: could not make directory {path} for some reason')
    return

def main():
    input_path = '/data/infinity-mirror/stats/deltacon'
    output_path = '/data/infinity-mirror/stats/deltacon-inverse'

    mkdir_output(output_path)

    for df, filename in load_df(input_path):
        for column in df:
            if column != 'model' and column != 'gen' and column != 'trial':
                df[column] = df[column].apply(lambda x: 1 - (1/(1 + x)))
        df.to_csv(f'{output_path}/{filename}', float_format='%.7f', sep='\t', index=False, na_rep='nan')
        print(f'wrote: {output_path}/{filename}')

    return

main()
