import os
import pickle

import scipy.sparse as sp

import src.netgan.netgan.utils as utils
from src.netgan.netgan.netgan import NetGAN

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import numpy as np
from sys import argv
import networkx as nx

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


def fit(adj):
    '''does the thing
        parameters:
            adj (scipy sparse csr): adjacency matrix for the input graph
        output:
            model (?): the trained model
    '''
    tf.reset_default_graph()
    lcc = utils.largest_connected_components(adj)
    adj = adj[lcc, :][:, lcc]
    n = adj.shape[0]

    val_share = 0.1
    test_share = 0.05

    print(f'n: {adj.shape[0]} m: {adj.sum() // 2}')

    # split the graph into train/test/validation
    max_tries = 10
    tries = 0
    while tries < max_tries:
        if tries == max_tries:
            raise Exception('NetGAN fit failed.. no split found')
        try:
            train_ones, val_ones, \
            val_zeros, test_ones, test_zeros = utils.train_val_test_split_adjacency(adj, val_share, test_share, undirected=True,
                                                                                    connected=True, asserts=True)
        except Exception as e:
            tries += 1
            print(f'Trying train test split again\n')
            continue
        break

    # set connected=False to prevent the MST business
    print(f'n: {adj.shape[0]} m: {adj.sum() // 2}, tr1: {len(train_ones)}, v0: {len(val_zeros)}, v1: {len(val_ones)}, te0: {len(test_zeros)}, te1: {len(test_ones)}')

    # generate the training graph and ensure it is symmetric
    train_graph = sp.coo_matrix((np.ones(len(train_ones)), (train_ones[:, 0], train_ones[:, 1]))).tocsr()
    assert (train_graph.toarray() == train_graph.toarray().T).all()

    rw_len = 16
    batch_size = 128

    walker = utils.RandomWalker(train_graph, rw_len, p=1, q=1, batch_size=batch_size)

    # define the model
    model = NetGAN(n, rw_len, walk_generator=walker.walk, gpu_id=0, use_gumbel=True, disc_iters=3,
                   W_down_generator_size=128, W_down_discriminator_size=128, l2_penalty_generator=1e-7,
                   l2_penalty_discriminator=5e-5, generator_layers=[40], discriminator_layers=[30],
                   temp_start=5, learning_rate=0.0003)

    # model = NetGAN(n, rw_len, walk_generator= walker.walk, gpu_id=0, use_gumbel=True, disc_iters=3,
    #                 W_down_discriminator_size=32, W_down_generator_size=128, l2_penalty_generator=1e-7,
    #                l2_penalty_discriminator=5e-5, generator_layers=[40], discriminator_layers=[30],
    #                temp_start=5, temperature_decay=0.99998, learning_rate=0.0003, legacy_generator=True)

    # stopping criterion can be one of 'val' or 'eo'
    stopping_criterion = 'eo'
    if stopping_criterion == 'eo':
        stopping = 0.5
    else:
        stopping = None

    eval_every = 750
    max_iters = 200_000

    # train the model
    model.train(A_orig=adj, val_ones=val_ones, val_zeros=val_zeros, stopping=stopping, eval_every=eval_every,
                max_patience=5, max_iters=max_iters)

    sample_walks = model.generate_discrete(10_000, reuse=True)

    samples = []
    for x in range(60):
        samples.append(sample_walks.eval({model.tau: 0.5}))

    model.session.close()  # close the interactive session to free up resources

    random_walks = np.array(samples).reshape([-1, rw_len])
    scores_matrix = utils.score_matrix_from_random_walks(random_walks, n).tocsr()

    return scores_matrix, train_graph.sum()


def main():
    if len(argv) < 2:
        print('Needs graph name and path to edgelist')
        exit(1)

    gname, path = argv[1:]

    g = nx.read_edgelist(path, create_using=nx.Graph(), nodetype=int)
    adj = nx.to_scipy_sparse_matrix(g)
    scores, tg_sum = fit(adj)

    pickle.dump((scores, tg_sum), open(f'./src/netgan/dumps/{gname}.pkl.gz', 'wb'))


if __name__ == '__main__':
    main()
