#!/usr/bin/env python
# coding: utf-8

# In[1]:


from netgan.netgan import *
import tensorflow as tf
from netgan import utils
import scipy.sparse as sp
import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import roc_auc_score, average_precision_score
import time

get_ipython().run_line_magic('matplotlib', 'inline')


# #### Load the data

# In[2]:


_A_obs, _X_obs, _z_obs = utils.load_npz('data/cora_ml.npz')
_A_obs = _A_obs + _A_obs.T
_A_obs[_A_obs > 1] = 1
lcc = utils.largest_connected_components(_A_obs)
_A_obs = _A_obs[lcc,:][:,lcc]
_N = _A_obs.shape[0]


# In[3]:


val_share = 0.1
test_share = 0.05
seed = 481516234


# #### Load the train, validation, test split from file

# In[4]:


loader = np.load('pretrained/cora_ml/split.npy').item()


# In[5]:


train_ones = loader['train_ones']
val_ones = loader['val_ones']
val_zeros = loader['val_zeros']
test_ones = loader['test_ones']
test_zeros = loader['test_zeros']


# In[6]:


train_graph = sp.coo_matrix((np.ones(len(train_ones)),(train_ones[:,0], train_ones[:,1]))).tocsr()
assert (train_graph.toarray() == train_graph.toarray().T).all()


# #### Parameters

# In[7]:


rw_len = 16
batch_size = 128


# In[8]:


walker = utils.RandomWalker(train_graph, rw_len, p=1, q=1, batch_size=batch_size)


# #### Create our NetGAN model

# In[9]:


netgan = NetGAN(_N, rw_len, walk_generator= walker.walk, gpu_id=3, use_gumbel=True, disc_iters=3,
                W_down_discriminator_size=32, W_down_generator_size=128,
                l2_penalty_generator=1e-7, l2_penalty_discriminator=5e-5,
                generator_layers=[40], discriminator_layers=[30], temp_start=5, temperature_decay=0.99998, learning_rate=0.0003, legacy_generator=True)


# #### Load pretrained model

# In[10]:


saver = tf.train.Saver()
saver.restore(netgan.session, "pretrained/cora_ml/pretrained_gen.ckpt")


# #### Generate random walks on the trained model

# In[11]:


sample_many = netgan.generate_discrete(10000, reuse=True, legacy=True)


# In[12]:


samples = []


# In[13]:


for _ in range(60):
    if (_+1) % 500 == 0:
        print(_+1)
    samples.append(sample_many.eval({netgan.tau: 0.5}))


# #### Assemble score matrix from the random walks

# In[14]:


rws = np.array(samples).reshape([-1, rw_len])
scores_matrix = utils.score_matrix_from_random_walks(rws, _N).tocsr()


# #### Compute graph statistics

# In[15]:


A_select = sp.csr_matrix((np.ones(len(train_ones)), (train_ones[:,0], train_ones[:,1])))


# In[16]:


A_select = train_graph


# In[17]:


sampled_graph = utils.graph_from_scores(scores_matrix, A_select.sum())
plt.spy(sampled_graph, markersize=.2)
plt.show()


# In[18]:


plt.spy(A_select, markersize=.2)
plt.show()


# In[19]:


utils.edge_overlap(A_select.toarray(), sampled_graph)/A_select.sum()


# In[20]:


utils.compute_graph_statistics(sampled_graph)


# In[21]:


utils.compute_graph_statistics(A_select.toarray())

