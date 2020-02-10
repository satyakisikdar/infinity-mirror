import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Train on CPU (hide GPU) due to memory constraints
os.environ['CUDA_VISIBLE_DEVICES'] = "0"

import tensorflow as tf
import numpy as np
import scipy.sparse as sp

from collections import namedtuple

from src.gae.gae.optimizer import OptimizerAE, OptimizerVAE
from src.gae.gae.model import GCNModelAE, GCNModelVAE
from src.gae.gae.preprocessing import preprocess_graph, construct_feed_dict, sparse_to_tuple, mask_test_edges

# Settings
flags = namedtuple('FLAGS', 'learning_rate epochs hidden1 hidden2 weight_decay dropout model dataset features')
FLAGS = flags(0.01, 200, 32, 16, 0., 0., 'gcn_ae', 'cora', 1)

# flags = tf.app.flags
# FLAGS = flags.FLAGS
# flags.DEFINE_float('learning_rate', 0.01, 'Initial learning rate.')
# flags.DEFINE_integer('epochs', 200, 'Number of epochs to train.')
# flags.DEFINE_integer('hidden1', 32, 'Number of units in hidden layer 1.')
# flags.DEFINE_integer('hidden2', 16, 'Number of units in hidden layer 2.')
# flags.DEFINE_float('weight_decay', 0., 'Weight for L2 loss on embedding matrix.')
# flags.DEFINE_float('dropout', 0., 'Dropout rate (1 - keep probability).')
#
# flags.DEFINE_string('model', 'gcn_ae', 'Model string.')
# flags.DEFINE_string('dataset', 'cora', 'Dataset string.')
# flags.DEFINE_integer('features', 1, 'Whether to use features (1) or not (0).')

def fit_ae(adj_matrix, epochs=200):
    ''' trains a non-variational graph autoencoder on a given input graph
        parameters:
            adj_matrix (ndarray):   adjacency matrix of the graph
            epochs (int):           how many iterations to train the model for
        output:
            a matrix containing probabilities corresponding to edges in an adjacency matrix
    '''
    # load data
    adj = adj_matrix
    features = sp.identity(adj.shape[0])

    # store original adjacency matrix (without diagonal entries) for later
    adj_orig = adj
    adj_orig = adj_orig - sp.dia_matrix((adj_orig.diagonal()[np.newaxis, :], [0]), shape=adj_orig.shape)
    adj_orig.eliminate_zeros()

    # compute train/test/validation splits
    while True:
        try:
            adj_train, train_edges, val_edges, val_edges_false, test_edges, test_edges_false = mask_test_edges(adj)
        except AssertionError as e:
            continue
        else:
            break
    adj = adj_train

    # some preprocessing
    adj_norm = preprocess_graph(adj)

    # define placeholders
    placeholders = {
        'features': tf.sparse_placeholder(tf.float32),
        'adj': tf.sparse_placeholder(tf.float32),
        'adj_orig': tf.sparse_placeholder(tf.float32),
        'dropout': tf.placeholder_with_default(0., shape=())
    }

    features = sparse_to_tuple(features.tocoo())
    num_features = features[2][1]
    features_nonzero = features[1].shape[0]

    # define the model
    model = GCNModelAE(placeholders, num_features, features_nonzero)

    pos_weight = float(adj.shape[0] * adj.shape[0] - adj.sum()) / adj.sum()
    norm = adj.shape[0] * adj.shape[0] / float((adj.shape[0] * adj.shape[0] - adj.sum()) * 2)

    # define the optimizer
    with tf.name_scope('optimizer'):
        opt = OptimizerAE(preds=model.reconstructions,
                          labels=tf.reshape(tf.sparse_tensor_to_dense(placeholders['adj_orig'], validate_indices=False), [-1]),
                          pos_weight=pos_weight,
                          norm=norm)

    # start up TensorFlow session
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    adj_label = adj_train + sp.eye(adj_train.shape[0])
    adj_label = sparse_to_tuple(adj_label)

    # train the model
    for epoch in range(epochs):
        # construct feed dictionary
        feed_dict = construct_feed_dict(adj_norm, adj_label, features, placeholders)
        feed_dict.update({placeholders['dropout']: FLAGS.dropout})

        # run single weight update
        outs = sess.run([opt.opt_op, opt.cost, opt.accuracy, opt.preds_sub], feed_dict=feed_dict)

    probs = sess.run(tf.nn.sigmoid(outs[3])).reshape(adj_matrix.shape)
    sess.close()
    return probs

def fit_vae(adj_matrix, epochs=200):
    ''' trains a variational graph autoencoder on a given input graph
        parameters:
            adj_matrix (ndarray):   adjacency matrix of the graph
            epochs (int):           how many iterations to train the model for
        output:
            a matrix containing probabilities corresponding to edges in an adjacency matrix
    '''
    # load data
    adj = adj_matrix
    features = sp.identity(adj.shape[0])

    # store original adjacency matrix (without diagonal entries) for later
    adj_orig = adj
    adj_orig = adj_orig - sp.dia_matrix((adj_orig.diagonal()[np.newaxis, :], [0]), shape=adj_orig.shape)
    adj_orig.eliminate_zeros()

    # compute train/test/validation splits
    while True:
        # compute train/test/validation splits
        try:
            adj_train, train_edges, val_edges, val_edges_false, test_edges, test_edges_false = mask_test_edges(adj)
        except AssertionError as e:
            continue
        else:
            break
    adj = adj_train

    # some preprocessing
    adj_norm = preprocess_graph(adj)

    # define placeholders
    placeholders = {
        'features': tf.sparse_placeholder(tf.float32),
        'adj': tf.sparse_placeholder(tf.float32),
        'adj_orig': tf.sparse_placeholder(tf.float32),
        'dropout': tf.placeholder_with_default(0., shape=())
    }

    num_nodes = adj.shape[0]
    features = sparse_to_tuple(features.tocoo())
    num_features = features[2][1]
    features_nonzero = features[1].shape[0]

    # define the model
    model = GCNModelVAE(placeholders, num_features, num_nodes, features_nonzero)

    pos_weight = float(adj.shape[0] * adj.shape[0] - adj.sum()) / adj.sum()
    norm = adj.shape[0] * adj.shape[0] / float((adj.shape[0] * adj.shape[0] - adj.sum()) * 2)

    # define the optimizer
    with tf.name_scope('optimizer'):
        opt = OptimizerVAE(preds=model.reconstructions,
                           labels=tf.reshape(tf.sparse_tensor_to_dense(placeholders['adj_orig'], validate_indices=False), [-1]),
                           model=model, num_nodes=num_nodes,
                           pos_weight=pos_weight,
                           norm=norm)

    # start up TensorFlow session
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    adj_label = adj_train + sp.eye(adj_train.shape[0])
    adj_label = sparse_to_tuple(adj_label)

    # train the model
    for epoch in range(epochs):
        # construct feed dictionary
        feed_dict = construct_feed_dict(adj_norm, adj_label, features, placeholders)
        feed_dict.update({placeholders['dropout']: FLAGS.dropout})

        # run single weight update
        outs = sess.run([opt.opt_op, opt.cost, opt.accuracy, opt.preds_sub], feed_dict=feed_dict)

    probs = sess.run(tf.nn.sigmoid(outs[3])).reshape(adj_matrix.shape)
    sess.close()
    return probs
