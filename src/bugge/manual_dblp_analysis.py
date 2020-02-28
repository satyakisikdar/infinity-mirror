import networkx as nx
from src.bugge.test_utils import *

G = nx.read_adjlist("graphs/dblp_cite.edge_list", create_using=nx.DiGraph, nodetype=int)
G = nx.DiGraph(G)
remove_self_loops(G)

single_citation_articles = set(filter(lambda n: G.in_degree(n) == 1, G.nodes()))
no_citation_articles = set(filter(lambda n: G.in_degree(n) == 0, G.nodes()))

out_neighbors = {n: set([j for (i, j) in G.out_edges(n)]) for n in G.nodes()}
in_neighbors = {n: set([i for (i, j) in G.in_edges(n)]) for n in G.nodes()}

single_citations_that_are_cited_by_no_citations = set(filter(lambda n: len(in_neighbors[n] & no_citation_articles) == 1, single_citation_articles))
any_citations_that_are_cited_by_no_citations = set(filter(lambda n: len(in_neighbors[n] & no_citation_articles) >= 1, G.nodes()))

print("Single citation articles (cited once) that are cited by no-citation articles: %s" % len(single_citations_that_are_cited_by_no_citations))
print("Single citation articles: %s" % len(single_citation_articles))

print("\n")

print("Num articles cited by at least one no-citation article: %s" % len(any_citations_that_are_cited_by_no_citations))
print("Num articles with at least on citation: %s" % (len(list(G.nodes())) - len(no_citation_articles)))

# Hmmm....

print("\n")

all_that_cite_singles = set(filter(lambda n: len(out_neighbors[n] & single_citation_articles) >= 1, G.nodes()))
no_citations_that_cite_singles = no_citation_articles & all_that_cite_singles

print("Number of no-citation articles that cite single-citation articles: %s" % len(no_citations_that_cite_singles))
print("Number of articles that cite single-citation articles: %s" % len(all_that_cite_singles))

# More Hmmmm...

print("\n")

count_of_nones_citations = sum([G.out_degree(n) for n in no_citation_articles])
count_of_all_citations = len(G.edges())

count_of_nones_citations_that_are_singles = len(single_citations_that_are_cited_by_no_citations)
count_of_all_citations_that_are_singles = len(single_citation_articles)

print("Percent of non-cited articles' citations that are articles with single citations: %s" % (count_of_nones_citations_that_are_singles / float(count_of_nones_citations)))
print("Percent of any articles' citations that are articles with single citation: %s" % (count_of_all_citations_that_are_singles / float(count_of_all_citations)))

print("\n")

out_equal = 0.0
out_distinct = 0.0
out_distinct_by_degree = 0.0
out_similar = 0.0
in_equal = 0.0
in_distinct = 0.0
in_distinct_by_degree = 0.0
in_similar = 0.0
total = 0.0

nodes = list(G.nodes())
for node_a in nodes:
    for node_b in (out_neighbors[node_a] | in_neighbors[node_a]):
        if node_a > node_b:
            continue
        out_neighbors_a = out_neighbors[node_a] - set([node_b])
        out_neighbors_b = out_neighbors[node_b] - set([node_a])
        in_neighbors_a = in_neighbors[node_a] - set([node_b])
        in_neighbors_b = in_neighbors[node_b] - set([node_a])
        out_overlap = out_neighbors_a & out_neighbors_b
        out_union = out_neighbors_a | out_neighbors_b
        out_difference = out_union - out_overlap
        in_overlap = in_neighbors_a & in_neighbors_b
        in_union = in_neighbors_a | in_neighbors_b
        in_difference = in_union - in_overlap
        if len(out_difference) == len(out_overlap):
            out_equal += 1.0
        elif len(out_difference) < len(out_overlap):
            out_similar += 1.0
        else:
            if abs(len(out_neighbors_a) - len(out_neighbors_b)) > min(len(in_neighbors_a), len(in_neighbors_b)):
                out_distinct_by_degree += 1.0
            else:
                out_distinct += 1.0

        if len(in_difference) == len(in_overlap):
            in_equal += 1.0
        elif len(in_difference) < len(in_overlap):
            in_similar += 1.0
        else:
            if abs(len(in_neighbors_a) - len(in_neighbors_b)) > min(len(in_neighbors_a), len(in_neighbors_b)):
                in_distinct_by_degree += 1.0
            else:
                in_distinct += 1.0
        total += 1.0

total /= 100.0

print("Neighbors Citing Both Articles:    Similar %.2f | Equal %.2f | Different %.2f | Different by degree %.2f" % \
    (out_similar / total, out_equal / total, out_distinct / total, out_distinct_by_degree / total))
print("Neighbors Cited By Both Articles:  Similar %.2f | Equal %.2f | Different %.2f | Different by degree %.2f" % \
    (in_similar / total, in_equal / total, in_distinct / total, in_distinct_by_degree / total))

print("Without difference by degree:")

total = (out_similar + out_equal + out_distinct) / 100.0
print("Neighbors Citing Both Articles:    Similar %.2f | Equal %.2f | Different %.2f" % \
    (out_similar / total, out_equal / total, out_distinct / total))
total = (in_similar + in_equal + in_distinct) / 100.0
print("Neighbors Cited By Both Articles:  Similar %.2f | Equal %.2f | Different %.2f" % \
    (in_similar / total, in_equal / total, in_distinct / total))

"""

print("\n")

print("For a Barabasi Albert Graph...")

G2 = nx.barabasi_albert_graph(len(G.nodes()), 4)
G3 = nx.DiGraph()
for node in G2.nodes():
    G3.add_node(node)
for edge in G2.edges():
    G3.add_edge(max(edge[0], edge[1]), min(edge[0], edge[1]))

single_citation_articles = set(filter(lambda n: G3.in_degree(n) == 1, G3.nodes()))
no_citation_articles = set(filter(lambda n: G3.in_degree(n) == 0, G3.nodes()))

single_citations_that_are_cited_by_no_citations = set(filter(lambda n: len(set([i for i, j in G3.in_edges(n)]) & no_citation_articles) == 1, single_citation_articles))
any_citations_that_are_cited_by_no_citations = set(filter(lambda n: len(set([i for i, j in G3.in_edges(n)]) & no_citation_articles) >= 1, G3.nodes()))

print("Single citation articles that are cited by no-citation articles: %s" % len(single_citations_that_are_cited_by_no_citations))
print("Single citation articles: %s" % len(single_citation_articles))

print("\n")

print("Num articles cited by at least one no-citation article: %s" % len(any_citations_that_are_cited_by_no_citations))
print("Num articles with at least on citation: %s" % (len(list(G.nodes())) - len(no_citation_articles)))

# Hmmm....

print("\n")

all_that_cite_singles = set(filter(lambda n: len(set([j for i, j in G3.out_edges(n)]) & single_citation_articles) >= 1, G3.nodes()))
no_citations_that_cite_singles = no_citation_articles & all_that_cite_singles

print("Number of no-citation articles that cite single-citation articles: %s" % len(no_citations_that_cite_singles))
print("Number of articles that cite single-citation articles: %s" % len(all_that_cite_singles))

# More Hmmmm...

print("\n")

count_of_nones_citations = sum([G3.out_degree(n) for n in no_citation_articles])
count_of_all_citations = len(G3.edges())

count_of_nones_citations_that_are_singles = len(single_citations_that_are_cited_by_no_citations)
count_of_all_citations_that_are_singles = len(single_citation_articles)

print("Percent of non-cited articles' citations that are articles with single citations: %s" % (count_of_nones_citations_that_are_singles / float(count_of_nones_citations)))
print("Percent of any articles' citations that are articles with single citation: %s" % (count_of_all_citations_that_are_singles / float(count_of_all_citations)))
"""
