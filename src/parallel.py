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
    x, y, z = args
    time.sleep(x // 5)
    return x + y + z

# todo test this with GraphStats
def parallel_imap(func, args, num_workers=10):
    results = []
    with Pool(num_workers) as pool:
        for result in pool.imap(func, args, chunksize=len(args)//num_workers):
            results.append(result)

    return results

def parallel_async(func, args, num_workers=10):
    def write_result(result):
        print(result)
        return result

    results = []
    async_promises = []
    with Pool(num_workers) as pool:
        for arg in args:
            r = pool.apply_async(func, [arg], callback=write_result)
            async_promises.append(r)
        for r in async_promises:
            try:
                r.wait()
                results.append(r.get())
            except Exception as e:
                results.append(r.get())

    return results

def sequential(func, args):
    results = []
    for arg in args:
        result = func(arg)
        results.append(result)
    return results

#@timer
#def main():
#    N = 100
#    args = [(x, y, z) for (x, y, z) in zip(range(N), range(N), range(N))]
#
#    #results = parallel_imap(doesathing, args)
#    results = parallel_async(doesathing, args)
#    #print(f'SUCCESS: there are {len(results)} results')
#    print(results)
#    return
