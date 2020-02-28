import networkx as nx
from src.bugge.full_approximate_rule_miner import *

G = nx.DiGraph()
for i in range(0, 6):
    G.add_node(i)
G.add_edge(0, 1)
G.add_edge(1, 2)
G.add_edge(2, 3)
G.add_edge(3, 4)

G.add_edge(1, 3)
G.add_edge(5, 3)

rm = FullApproximateRuleMiner(G, 2, 2, None)

while not rm.done():
    best_rule = rm.determine_best_rule()
    rm.contract_valid_tuples(best_rule)
rm.cost_comparison()
