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
            if 'csv' in filename:
                print(f'\tloading {subdir} {filename} ... ', end='', flush=True)
                df = pd.read_csv(os.path.join(subdir, filename), sep=','), filename
                print('done')
                yield df

def compute_stats(densities):
    #padding = max(len(l) for l in js)
    #for idx, l in enumerate(js):
    #    while len(js[idx]) < padding:
    #        js[idx] += [np.NaN]
    for idx, l in enumerate(densities):
        if l == []:
            densities[idx] = [0]
    print(densities)
    mean = np.nanmean(densities, axis=0)
    ci = []
    for row in np.asarray(densities).T:
        ci.append(st.t.interval(0.95, len(row)-1, loc=np.mean(row), scale=st.sem(row)))
    return np.asarray(mean), np.asarray(ci)

def construct_table(abs_densities, seq_densities, model):
    abs_mean, abs_ci = compute_stats(abs_densities)
    seq_mean, seq_ci = compute_stats(seq_densities)
    gen = [x + 1 for x in range(len(abs_mean))]

    rows = {'model': model, 'gen': gen, 'abs_mean': abs_mean, 'abs-95%': abs_ci[:,0], 'abs+95%': abs_ci[:,1], 'seq_mean': seq_mean, 'seq-95%': seq_ci[:,0], 'seq+95%': seq_ci[:,1]}

    df = pd.DataFrame(rows)
    return df

def abs_mean(a):
    return np.mean(a)

def abs95u(a):
    return st.t.interval(0.95, len(a) - 1, loc=np.mean(a), scale=st.sem(a))[1]

def abs95d(a):
    return st.t.interval(0.95, len(a) - 1, loc=np.mean(a), scale=st.sem(a))[0]

def main():
    base_path = '/data/infinity-mirror/stats/embedding'
    output_path = '/home/dgonza26/infinity-mirror/data/netlsd'

    for df, filename in load_df(base_path):
        # compute the differences with respect to G0
        G0 = df.loc[df['level'] == 0] #.drop(['model', 'gen', 'trial'], axis=1)
        for column in df:
            if column not in ['name', 'level', 'model']:
                df[column] = (df[column] - G0[column].loc[0])**2
                #df[column] = df[column].apply(lambda x: np.sqrt(np.sum(x**2)))
        #print(df)
        #print(df.drop(['trial', 'gen', 'model'], axis=1).loc[0] - G0.loc[0])

        #for column in df.drop(['trial', 'model', 'gen'], axis=1):
        #    print(graphlet_counts[column])
        #print(graphlet_counts)

        # rename column
        df['gen'] = df['level']
        df = df.drop(['level'], axis=1)

        # compute Euclidean distance
        df['dist'] = df.drop(['name', 'gen', 'model'], axis=1).sum(axis=1)
        df['dist'] = df['dist'].apply(lambda x: np.sqrt(x))

        # drop extra columns
        for column in df.columns:
            if column not in ['model', 'gen', 'dist']:
                df = df.drop(column, axis=1)

        df = df.groupby(['model', 'gen']).agg([abs_mean, abs95d, abs95u])
        df.columns = df.columns.droplevel(0)
        df = df.drop(df.index[0])

        # to reset the columns, write then read again
        df.to_csv(f'./temp.csv', float_format='%.7f', sep='\t', na_rep='nan')
        df = pd.read_csv(f'./temp.csv', sep='\t')

        # normalize the columns by the sum of the column
        t = df.drop(['gen', 'model'], axis=1)
        t = t.div(t.sum(axis=0), axis=1)
        df['abs_mean'] = t['abs_mean']
        df['abs95d'] = t['abs95d']
        df['abs95u'] = t['abs95u']

        # thresholding with 0.0001
        for column in df:
            if column not in ['trial', 'gen', 'model']:
                df[column] = df[column].apply(lambda x: 0.0001 if x < 0.0001 else x)

        df.to_csv(f'{output_path}/{filename}', float_format='%.7f', sep='\t', na_rep='nan')
        print(f'wrote: {output_path}/{filename}')

    #df.to_csv(f'{output_path}/{dataset}_{model}_density.csv', float_format='%.7f', sep='\t', index=False, na_rep='nan')

    return

main()
