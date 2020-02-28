import networkx as nx
import argparse
import os
from src.bugge.full_approximate_rule_miner import *
from src.bugge.test_utils import *
from src.bugge.generation import *

parser = argparse.ArgumentParser()
parser.add_argument("rule_min", type=int, help="Minimum rule size")
parser.add_argument("rule_max", type=int, help="Maximum rule size")
parser.add_argument("graph_type", help="The graph type. Either the path to an edgelist file or one of the following: watts_strogats, n_tree, and n_tree_of_k_rings")
parser.add_argument("-v", "--vertices", type=int, help="The number of nodes in the graph.")
parser.add_argument("-s", "--shortcut", type=int, help="A parameter to make the code run faster. 0 is fastest, but low numbers may result in finding fewer good rules.")
parser.add_argument("-n", type=int, help="A graph parameter. For watts_strogats, specifies number of neighbors. For the trees, specifies number of children.")
parser.add_argument("-k", type=int, help="A graph parameter. For n_tree_of_k_rings, specifies the ring size.")
parser.add_argument("-r", type=float, help="Probability that an edge gets rewired.")
parser.add_argument("--bidirected", help="Add if the graph should be bidirected. Currently just for watts_strogatz")
parser.add_argument("--degree_copy", help="Runs on a graph with the same degree distribution as the specified graph.")
parser.add_argument("--export_edge_list", help="A file to which the edge list of the generated graph should be exported.")
args = parser.parse_args()
if args.graph_type == "watts_strogatz":
    if args.k:
        print("INFO: k was specified but is not used in watts_strogatz")
    if not args.n:
        print("ERROR: need parameter n for watts_strogatz. Aborting.")
        exit(2)
if args.graph_type == "n_tree":
    if args.k:
        print("INFO: k was specified but is not used in an n_tree")
    if not args.n:
        print("ERROR: need parameter n for n_tree. Aborting.")
        exit(2)
if args.graph_type == "n_tree_of_k_rings":
    if not args.k:
        print("ERROR: need k for n_tree_of_k_rings. Aborting.")
        exit(2)
    if not args.n:
        print("ERROR: need parameter n for n_tree_of_k_rings. Aborting.")
        exit(2)
if args.graph_type not in ["watts_strogatz", "n_tree", "n_tree_of_k_rings"]:
    if not os.path.isfile(args.graph_type):
        print("ERROR: file [%s] not found. Aborting." % args.graph_type)
        exit(2)
if args.r is not None and (args.r < 0.0 or 1.0 < args.r):
    print("ERROR: rewiring probability r must be in the range [0.0, 1.0]. Aborting.")
    exit(2)

G = None
if args.graph_type == "n_tree":
    G = n_ary_tree(args.vertices, args.n) 
    G = relabel_nodes(G)
elif args.graph_type == "n_tree_of_k_rings":
    G = n_ary_tree_of_k_rings(args.vertices, args.n, args.k)
    G = relabel_nodes(G)
elif args.graph_type == "watts_strogatz":
    bidirected = args.bidirected is not None
    G = watts_strogatz(args.vertices, args.n, bidirected)
    G = relabel_nodes(G)
else:
    G = nx.read_adjlist(args.graph_type, create_using=nx.DiGraph, nodetype=int)
    G = nx.DiGraph(G)
    remove_self_loops(G)
    bidirected = 0.0
    connections = 0.0
    for edge in G.edges():
        connections += 1.0
        if edge[0] < edge[1] and ((edge[1], edge[0]) in G.edges()):
            bidirected += 1.0
            connections -= 1.0
    print("Percent of connections that are bidirected: %s" % (100.0 * bidirected / connections))

if args.r is not None:
    rewire_graph(G, args.r)

if args.degree_copy:
    node_list = list(G.nodes())
    in_degs = [G.in_degree(node) for node in node_list]
    out_degs = [G.out_degree(node) for node in node_list]
    G = nx.DiGraph(nx.directed_configuration_model(in_degs, out_degs))
    remove_self_loops(G)

if args.export_edge_list:
    nx.write_adjlist(G, args.export_edge_list)

while not rm.done():
    best_rule = rm.determine_best_rule()
    rm.contract_valid_tuples(best_rule)
rm.cost_comparison()
