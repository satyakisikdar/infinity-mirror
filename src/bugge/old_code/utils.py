import networkx as nx
from networkx import utils
#from networkx.algorithms.bipartite.generators import configuration_model
#from networkx.algorithms import isomorphism
#from networkx.algorithms.shortest_paths.unweighted import all_pairs_shortest_path_length
#from networkx.algorithms.components import is_connected
#import numpy as np



# Assumes at most 1024 nodes in G
def verify_two_nodes_to_id_works(G):
    checks = {}
    for i in range(0, len(G.nodes())):
        for j in range(0, len(G.nodes())):
            if i == j:
                continue
            nodes = [i, j]
            nodes.sort()
            id_from_ids = (nodes[0] << 10) + nodes[1]
            id_from_function = two_nodes_to_id(G, i, j)
            if id_from_ids in checks:
                if id_from_function != checks[id_from_ids][0]:
                    print("Error!")
                    print(id_from_function)
                    print(checks[id_from_ids])
                    print("i: " + str(checks[id_from_ids][1]) + " j: " + str(checks[id_from_ids][2]))
                    print("i': " + str(i) + " j': " + str(j))
            else:
                checks[id_from_ids] = (id_from_function, i, j)

# Assumes at most 1024 nodes in G
def verify_three_nodes_to_id_works(G):
    checks = {}
    for i in range(0, len(G.nodes())):
        for j in range(0, len(G.nodes())):
            for k in range(0, len(G.nodes())):
                if i == j or i == k or j == k:
                    continue
                nodes = [i, j, k]
                nodes.sort()
                id_from_ids = (nodes[0] << 20) + (nodes[1] << 10) + nodes[2]
                id_from_function = three_nodes_to_id(G, i, j, k)
                if id_from_ids in checks:
                    if id_from_function != checks[id_from_ids][0]:
                        print("Error!")
                        print(id_from_function)
                        print(checks[id_from_ids])
                        print("i: " + str(checks[id_from_ids][1]) + " j: " + str(checks[id_from_ids][2]) + " k: " + str(checks[id_from_ids][3]))
                        print("i': " + str(i) + " j': " + str(j) + " k': " + str(k))
                else:
                    checks[id_from_ids] = (id_from_function, i, j, k)

# Takes a graph and replaces a node with a three-rule:
def replace_node_with_three(G, node_id, three_id, next_id):
    in_edges = G.in_edges(node_id)
    out_edges = G.out_edges(node_id)
    if not three_id & (1 << 11): # If first node has no incoming edges:
        for edge in in_edges:
            G.remove_edge(edge[0], edge[1])
    if not three_id & (1 << 10): # If first node has no outgoing edges:
        for edge in out_edges:
            G.remove_edge(edge[0], edge[1])
    G.add_node(next_id)
    if three_id & (1 << 9): # If second node has incoming edges:
        for edge in in_edges:
            G.add_edge(edge[0], next_id)
    if three_id & (1 << 8): # If second node has outgoing edges:
        for edge in out_edges:
            G.add_edge(next_id, edge[1])
    next_id += 1
    if three_id & (1 << 7): # If third node has incoming edges:
        for edge in in_edges:
            G.add_edge(edge[0], next_id)
    if three_id & (1 << 6): # If third node has outgoing edges:
        for edge in out_edges:
            G.add_edge(next_id, edge[1])

    # TODO: Add code for connecting the three nodes.

G=nx.scale_free_graph(100)
verify_two_nodes_to_id_works(G)
verify_three_nodes_to_id_works(G)