from src.netgan.netgan import *
from src.netgan.netgan import utils

from sklearn.metrics import roc_auc_score, average_precision_score
import scipy.sparse as sp
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import numpy as np
import time

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

def fit(adj):
    '''does the thing
        parameters:
            adj (scipy sparse csr): adjacency matrix for the input graph
        output:
            model (?): the trained model
    '''
    lcc = utils.largest_connected_components(adj)
    adj = adj[lcc,:][:,lcc]
    n = adj.shape[0]

    val_share = 0.1
    test_share = 0.05

    # split the graph into train/test/validation
    train_ones, val_ones, val_zeros, test_ones, test_zeros = utils.train_val_test_split_adjacency(adj, val_share, test_share, undirected=True, connected=True, asserts=True)

    # generate the training graph and ensure it is symmetric
    train_graph = sp.coo_matrix((np.ones(len(train_ones)),(train_ones[:, 0],train_ones[:,1]))).tocsr()
    assert (train_graph.toarray() == train_graph.toarray().T).all()

    rw_len = 16
    batch_size = 128

    walker = utils.RandomWalker(train_graph, rw_len, p=1, q=1, batch_size=batch_size)

    # define the model
    model = NetGAN(n, rw_len, walk_generator=walker.walk, \
            gpu_id=0, use_gumbel=True, disc_iters=3, \
            W_down_generator_size=128, W_down_discriminator_size=128, \
            l2_penalty_generator=1e-7, l2_penalty_discriminator=5e-5, \
            generator_layers=[40], discriminator_layers=[30], \
            temp_start=5, learning_rate=0.0003)

    # stopping criterion can be one of 'val' or 'eo'
    stopping_criterion = 'val'
    if stopping_criterion == 'eo':
        stopping = 0.5
    else:
        stopping = None

    eval_every = 3
    #max_iters = 30000
    max_iters = 4

    # train the model
    print('hi')
    log_dict = model.train(A_orig=adj, val_ones=val_ones, val_zeros=val_zeros, \
            stopping=stopping, eval_every=eval_every, max_patience=5, max_iters=max_iters)
    print('hello')

    sample_walks = model.generate_discrete(10000, reuse=True)

    samples = []
    for x in range(60):
        samples.append(sample_walks.eval({model.tau: 0.5}))
        #if (x + 1) % 10 == 0:
        #    print(x + 1)

    random_walks = np.array(samples).reshape([-1, rw_len])
    scores_matrix = utils.score_matrix_from_random_walks(random_walks, n).tocsr()

    return scores_matrix, train_graph.sum()

def gen(scores, tg_sum):
    return utils.graph_from_scores(scores, tg_sum)
#
#def main():
#    A, _X_obs, _z_obs = utils.load_npz('data/cora_ml.npz')
#    A = A + A.T
#    A[A > 1] = 1
#
#    scores, tg_sum = fit(A)
#
#    sampled_graph = gen(scores, tg_sum)
#    print(sampled_graph)
#    np.savetxt('wew2.dat', sampled_graph, fmt='%d')
#
#    #print('--------------------------')
#    #print(scores)
#    #print(tg_sum)
#    #np.save('./scores', scores)
