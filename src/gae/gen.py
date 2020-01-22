import numpy as np
from networkx import Graph, from_numpy_array

def gen(probs):
    ''' generates an adjacency matrix based on a trained probability matrix
        parameters:
            probs (ndarray):        a matrix of probabilities for the edges
        output:
            generated (nx.Graph):   an undirected graph generated from the given probabilities
    '''
    n = probs.shape[0]
    sample = np.random.rand(n, n) <= probs
    sample = sample * sample
    np.fill_diagonal(sample, False)
    return nx.from_numpy_array(gen_mat, create_using=nx.Graph())
