import platform
import subprocess
import networkx as nx
import numpy as np
import pandas as pd
import scipy.spatial
import scipy.stats

from src.utils import check_file_exists, delete_files

np.seterr(all='ignore')


def GCD(h1, h2):
    assert h1.name != '' and h2.name != '', 'Graph names cannot be empty'
    df_g = external_orca(h1, f'{h1.name}_o')
    df_h = external_orca(h2, f'{h2.name}_t')

    gcm_g = tijana_eval_compute_gcm(df_g)
    gcm_h = tijana_eval_compute_gcm(df_h)

    gcd = tijana_eval_compute_gcd(gcm_g, gcm_h)
    return round(gcd, 3)


def external_orca(g: nx.Graph, gname: str):
    if not isinstance(g, nx.Graph):
        g = nx.Graph(g)  # convert it into a simple graph

    self_loop_edges = list(nx.selfloop_edges(g))
    if len(self_loop_edges) > 0:
        g.remove_edges_from(self_loop_edges)

    if nx.number_connected_components(g) > 1:
        g = g.subgraph(max(nx.connected_components(g), key=len))
    if nx.is_directed(g):
        selfloops = g.selfloop_edges()
        g.remove_edges_from(selfloops)   # removing self-loop edges

    g = nx.convert_node_labels_to_integers(g, first_label=0)

    file_dir = 'src/scratch'
    input_path = f'./{file_dir}/{gname}.in'
    with open(input_path, 'w') as f:
        f.write(f'{g.order()} {g.size()}\n')
        for u, v in g.edges():
            f.write(f'{u} {v}\n')

    args = ['', '4', f'./{file_dir}/{gname}.in', f'./{file_dir}/{gname}.out']

    if 'Windows' in platform.platform():
        args[0] = './src/orca.exe'
    elif 'Linux' in platform.platform():
        args[0] = './src/orca_linux'
    else:
        args[0] = './src/orca_mac'

    process = subprocess.run(' '.join(args), shell=True, stdout=subprocess.DEVNULL)
    if process.returncode != 0:
        print('Error in ORCA')

    output_path = f'./{file_dir}/{gname}.out'
    assert check_file_exists(output_path), f'output file @ {output_path} not found in GCD'
    df = pd.read_csv(output_path, sep=' ', header=None)

    # delete both the input and output files
    delete_files(input_path, output_path)

    return df


def spearmanr(x, y):
    """
    Spearman correlation - takes care of the nan situation
    :param x:
    :param y:
    :return:
    """
    score = scipy.stats.spearmanr(x, y)[0]
    if np.isnan(score):
        score = 1
    return score


def tijana_eval_compute_gcm(G_df):
    l = G_df.shape[1]  # no of graphlets: #cols in G_df

    M = G_df.values  # matrix of nodes & graphlet counts
    M = np.transpose(M)  # transpose to make it graphlet counts & nodes
    gcm = scipy.spatial.distance.squareform(   # squareform converts the sparse matrix to dense matrix
        scipy.spatial.distance.pdist(M,   # compute the pairwise distances in M
                                     spearmanr))  # using spearman's correlation
    gcm = gcm + np.eye(l, l)   # make the diagonals 1 (dunno why, but it did that in the original code)
    return gcm


def tijana_eval_compute_gcd(gcm_g, gcm_h):
    assert len(gcm_h) == len(gcm_g), "Correlation matrices must be of the same size"

    gcd = np.sqrt(   # sqrt
        np.sum(  # of the sum of elements
            (np.triu(gcm_g) - np.triu(gcm_h)) ** 2   # of the squared difference of the upper triangle values
        ))
    if np.isnan(gcd):
        print('GCD is nan')
    return round(gcd, 3)