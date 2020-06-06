import networkx as nx
from src.portrait.portrait_divergence import portrait_divergence as pd

g = nx.ring_of_cliques(500, 4)
h = nx.ring_of_cliques(500, 4)

print(pd(g, h))
