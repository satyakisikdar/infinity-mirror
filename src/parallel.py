import os
import sys
import time
import timeit
#import multiprocessing as mp
from multiprocessing import Pool

import numpy as np
import networkx as nx

from tqdm import tqdm

def timer(function):
    def new_function():
        start_time = timeit.default_timer()
        function()
        elapsed = timeit.default_timer() - start_time
        print(f'Function "{function.__name__}" took {elapsed} seconds to complete.')
    return new_function()

def doesathing(args):
    time.sleep(1)
    x, y, z = args
    return x + y + z

# todo test this with GraphStats
def parallel(func, args, num_workers=10):
    results = []
    print(len(args))

    with Pool(num_workers) as pool:
        for result in pool.imap(func, args, chunksize=len(args)//num_workers):
            results.append(result)

    return results

def sequential(func, args):
    results = []
    for arg in args:
        result = func(arg)
        results.append(result)
    return results
#@timer
def main():
    N = 100
    args = [(x, y, z) for (x, y, z) in zip(range(N), range(N), range(N))]

    results = sequential(doesathing, args)
    print(f'SUCCESS: there are {len(results)} results')
    return
