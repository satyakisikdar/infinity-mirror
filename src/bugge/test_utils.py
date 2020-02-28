import networkx as nx
import random
import numpy as np

def n_ary_tree(size, n):
    G=nx.DiGraph()
    for i in range(1, size + 1):
        G.add_node(i)
    for i in range(1, size + 1):
        for j in range(0, n):
            if i * n + j - (n - 2) < size + 1:
                G.add_edge(i, i * n + j - (n - 2))
    return G

def n_ary_tree_of_k_rings(size, n, k):
    tree_size = int(size / k)
    leftovers = size % k

    G = nx.DiGraph()
    for i in range(1, size):
        G.add_node(i)
    # Make the rings:
    for tree_idx in range(1, tree_size + 1):
        graph_idx = (tree_idx - 1) * k + 1
        for ring_idx in range(graph_idx, graph_idx + k - 1):
            G.add_edge(ring_idx, ring_idx + 1)
        G.add_edge(graph_idx + k - 1, graph_idx)
    # Make the tree:
    for tree_idx in range(1, tree_size + 1):
        graph_idx = (tree_idx - 1) * k + 1
        ring_bottom = graph_idx + int(k / 2)
        for j in range(0, n):
            next_tree_idx = tree_idx * n + j - (n - 2)
            next_graph_idx = (next_tree_idx - 1) * k + 1
            if next_graph_idx <= tree_size * k:
                G.add_edge(ring_bottom, next_graph_idx)
    return G

# Note that, due to the current coding, and rewiring_prob over 0.5 will cause all edges to be rewired.
def watts_strogatz(size, k, bidirected=True):
    G = nx.DiGraph()
    for i in range(0, size):
        G.add_node(i)
    for i in range(0, size):
        for j in range(1, k + 1):
            G.add_edge(i, (i + j) % size)
            if bidirected:
                G.add_edge((i + j) % size, i)
    return G

def rewire_graph(G, rewiring_prob):
    original_edges = set([edge for edge in G.edges()])
    rewired_edges = set()
    # Choose which edges to rewire and delete them from the graph.
    for edge in original_edges:
        if random.uniform(0.0, 0.999999999999999) < rewiring_prob:
            rewired_edges.add(edge)
            G.remove_edge(edge[0], edge[1])
    
    # Put the rewired edges back into the graph.
    final_edges = original_edges - rewired_edges
    nodes = list(G.nodes())
    for edge in rewired_edges:
        new_edge = (nodes[random.randint(0, len(nodes) - 1)], nodes[random.randint(0, len(nodes) - 1)])
        while new_edge[0] == new_edge[1] or new_edge in final_edges: # While there's a self-loop or a collision
            new_edge = (nodes[random.randint(0, len(nodes) - 1)], nodes[random.randint(0, len(nodes) - 1)])
        G.add_edge(new_edge[0], new_edge[1])
        final_edges.add(new_edge)

def remove_self_loops(G):
    for node in G.nodes():
        if (node, node) in G.edges():
            G.remove_edge(node, node)

def relabel_nodes(G):
    G_prime = nx.DiGraph()
    nodes = list(G.nodes())
    for node in nodes:
        G_prime.add_node(node)

    perm = np.random.permutation(len(nodes))
    old_to_new = {nodes[i]: nodes[perm[i]] for i in range(0, len(nodes))}
    for edge in G.edges():
        G_prime.add_edge(old_to_new[edge[0]], old_to_new[edge[1]])

    return G_prime
        
# This code is unfinished. It was copied from a place that used it for UNDIRECTED graphs.
def make_graph_with_same_degree_dist(G):
    G_sequence = list(d for n, d in G.degree())
    G_sequence.sort()
    sorted_G_sequence = list((d, n) for n, d in G.degree())
    sorted_G_sequence.sort(key=lambda tup: tup[0])
    done = False
    while not done:
        G_prime = nx.configuration_model(G_sequence)
        G_prime = nx.Graph(G_prime)
        G_prime.remove_edges_from(G_prime.selfloop_edges())
        tries = 10
        while tries > 0 and (len(G.edges()) != len(G_prime.edges())):
            sorted_G_prime_sequence = list((d, n) for n, d in G_prime.degree())
            sorted_G_prime_sequence.sort(key=lambda tup: tup[0])
            #print("Sorted G_sequence:")
            #print(sorted_G_sequence)
            #print("Sorted G_prime_sequence:")
            #print(sorted_G_prime_sequence)
            missing = []
            for i in range(0, len(G.nodes())):
                while sorted_G_sequence[i][0] > sorted_G_prime_sequence[i][0]:
                    missing.append(sorted_G_prime_sequence[i][1])
                    sorted_G_prime_sequence[i] = (sorted_G_prime_sequence[i][0] + 1, sorted_G_prime_sequence[i][1])
            missing = np.random.permutation(missing)
            if len(missing) % 2 != 0:
                print("Sanity issue! Alert!")
            #print("Edges before:")
            #print(G_prime.edges())
            #print("Missing:")
            #print(missing)
            for i in range(0, int(len(missing) / 2)):
                G_prime.add_edge(missing[2*i], missing[2*i + 1])
            G_prime = nx.Graph(G_prime)
            G_prime.remove_edges_from(G_prime.selfloop_edges())
            #print("Edges after:")
            #print(G_prime.edges())
            #if not is_connected(G_prime):
                #print("Bad: G_prime disconnected")
            tries -= 1
        if not is_connected(G_prime):
            pass
        elif len(G.edges()) == len(G_prime.edges()):
            #print("Graph creation successful")
            done = True
    return G_prime

def weird():
    G = nx.DiGraph()
    for i in range(0, 21):
        G.add_node(i)
    for block in [0, 7, 14]:
        G.add_edge(0 + block, 1 + block)
        G.add_edge(0 + block, 2 + block)
        G.add_edge(1 + block, 3 + block)
        G.add_edge(1 + block, 4 + block)
        G.add_edge(2 + block, 5 + block)
        G.add_edge(2 + block, 6 + block)
        for i in range(0, 7):
            for j in range(7, 21):
                G.add_edge(i + block, (i + block + j) % 21)
    return G
