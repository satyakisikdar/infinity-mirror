from src.gae.fit import fit_vae, fit_ae
import networkx as nx
import numpy as np


g = nx.cycle_graph(50)
mat = nx.adjacency_matrix(g)
probs = fit_ae(mat, epochs=200)

print(probs.T == probs)

target = np.zeros_like(mat)
N = mat.shape[0]


for i in range(20):
    rand_mat = np.random.rand(N, N)

    gen_mat = probs >= rand_mat
    gen_mat = np.logical_and(gen_mat, gen_mat.T)
    np.fill_diagonal(gen_mat, False)

    gen_g = nx.from_numpy_array(gen_mat, create_using=nx.Graph())
    print(i, gen_g.order(), gen_g.size())