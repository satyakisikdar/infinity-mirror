import networkx as nx
from networkx import utils

class Rule:
    """Stores a grammar rule from a graph and can apply itself to that graph"""

    def __init__(self, t, G, rule_lib=None):
        self._size = len(t)
        if self._size == 2:
            self._id = self._two_nodes_to_id(G, t[0], t[1])
        elif self._size == 3:
            self._id = self._three_nodes_to_id(G, t[0], t[1], t[2])
        else:
            if rule_lib is None:
                print("Error! Need rule lib.")
            # Build a graph and then get an id from the rule_lib.
            actual_graph = self._rule_graph(t, G)
            self._id = rule_lib.add_rule(actual_graph)

    def _rule_graph(self, t, G):
        actual_graph = nx.DiGraph()
        nodes = {}
        for n in t:
            nodes[n] = 1
            actual_graph.add_node(n)

        # Add nodes with self loops to mark in and out edges:
        actual_graph.add_node("__in__")
        actual_graph.add_node("__out__")
        actual_graph.add_edge("__in__", "__in__")
        actual_graph.add_edge("__out__", "__out__")
        for n in t:
            in_edges = G.in_edges(n)
            out_edges = G.out_edges(n)
            in_set = False
            for ie in in_edges:
                if ie[0] == n:
                    pass
                    # WE HAVE A SELF LOOP! DON'T KNOW HOW TO HANDLE THIS!
                elif ie[0] in nodes:
                    actual_graph.add_edge(ie[0], ie[1])
                elif not in_set:
                    actual_graph.add_edge("__in__", n)
                    in_set = True
            out_set = False
            for oe in out_edges:
                if oe[1] == n:
                    pass
                    # WE HAVE A SELF LOOP! DON'T KNOW HOW TO HANDLE THIS!
                elif oe[1] in nodes:
                    actual_graph.add_edge(oe[0], oe[1])
                elif not out_set:
                    actual_graph.add_edge(n, "__out__")
                    out_set = True
        return actual_graph

    def size(self):
        return self._size

    def equals(self, r):
        if r.size() != self._size:
            return False
        return r.id() == self._id

    def id(self):
        return self._id

    # Takes in a digraph and, handling isomorphisms, outputs a number with the following bits:
    # zabcdefghijkl
    # z: a bit set to 1 to indicate that this is a three-node id
    # a: Does node_0 have the external in-edges?
    # b: Does node_0 have the external out-edges?
    # c: Does node_1 have the external in-edges?
    # d: ...
    # e:
    # f:
    # g: Does node_0 point to node_1?
    # h: Does node_0 point to node_2?
    # i: Does node_1 point to node_0?
    # j: Does node_1 point to node_2?
    # k: Does node_2 point to node_0?
    # l: Does node_2 point to node_1?
    def _three_nodes_to_id(self, G, a, b, c):
        three_nodes = [{"id": a, "e_in": 0, "e_out": 0, "i_doub": 0, "i_in": 0, "i_out": 0},
                       {"id": b, "e_in": 0, "e_out": 0, "i_doub": 0, "i_in": 0, "i_out": 0},
                       {"id": c, "e_in": 0, "e_out": 0, "i_doub": 0, "i_in": 0, "i_out": 0}]
        # First we need to get an ordering of the nodes that is impervious to isomorphisms:
        for i in range(0, 3):  # O(3 if getting list of neighbors is free)
            for in_edge in G.in_edges(three_nodes[i]["id"]):  # O(3 if getting list of neighbors is free)
                if in_edge[0] != three_nodes[(i + 1) % 3]["id"] and in_edge[0] != three_nodes[(i + 2) % 3]["id"]:
                    three_nodes[i]["e_in"] = 1
                    break
            for out_edge in G.out_edges(three_nodes[i]["id"]):  # O(3 if getting list of neighbors is free)
                # print(str(three_nodes[i][0]) + " vs " + str(out_edge[1]))
                if out_edge[1] != three_nodes[(i + 1) % 3]["id"] and out_edge[1] != three_nodes[(i + 2) % 3]["id"]:
                    three_nodes[i]["e_out"] = 1
                    break
            for j in range(1, 3):
                in_j = G.has_edge(three_nodes[i]["id"], three_nodes[(i + j) % 3]["id"])
                out_j = G.has_edge(three_nodes[(i + j) % 3]["id"], three_nodes[i]["id"])
                if in_j:
                    three_nodes[i]["i_in"] += 1
                if out_j:
                    three_nodes[i]["i_out"] += 1
                if in_j and out_j:
                    three_nodes[i]["i_doub"] += 1

        three_nodes.sort(
            key=lambda x: (x["e_in"] << 7) + (x["e_out"] << 6) + (x["i_doub"] << 4) + (x["i_in"] << 2) + x["i_out"])

        final_number = (1 << 12) + \
                       (three_nodes[0]["e_in"] << 11) + (three_nodes[0]["e_out"] << 10) + \
                       (three_nodes[1]["e_in"] << 9) + (three_nodes[1]["e_out"] << 8) + \
                       (three_nodes[2]["e_in"] << 7) + (three_nodes[2]["e_out"] << 6)
        first = three_nodes[0]["id"]
        second = three_nodes[1]["id"]
        third = three_nodes[2]["id"]
        final_number += (int(G.has_edge(first, second)) << 5) + (int(G.has_edge(first, third)) << 4)
        final_number += (int(G.has_edge(second, first)) << 3) + (int(G.has_edge(second, third)) << 4)
        final_number += (int(G.has_edge(third, first)) << 1) + int(G.has_edge(third, second))
        return final_number

    # Takes in a digraph and, handling isomorphisms, outputs a number with the following bits:
    # abcdef
    # a: Does node_0 have the external in-edges?
    # b: Does node_0 have the external out-edges?
    # c: Does node_1 have the external in-edges?
    # d: Does node_1 have the external out-edges?
    # e: Does node_0 point to node_1?
    # f: Does node_1 point to node_0?
    def _two_nodes_to_id(self, G, a, b):
        two_nodes = [[a, 0, 0], [b, 0, 0]]
        # First, get their relationship to the outside world.
        # 2 indicates has external in-edges
        # 1 indicates has external out-edges
        for i in range(0, 2):  # O(2 if getting list of neighbors is free)
            for in_edge in G.in_edges(two_nodes[i][0]):  # O(2 if getting list of neighbors is free)
                if in_edge[0] != two_nodes[(i + 1) % 2][0]:
                    two_nodes[i][1] += 2
                    break
            for out_edge in G.out_edges(two_nodes[i][0]):  # O(2 if getting list of neighbors is free)
                # print(str(three_nodes[i][0]) + " vs " + str(out_edge[1]))
                if out_edge[1] != two_nodes[(i + 1) % 2][0]:
                    two_nodes[i][1] += 1
                    break
        for i in range(0, 2):
            if G.has_edge(two_nodes[i][0], two_nodes[(i + 1) % 2][0]):
                two_nodes[i][2] += 1

        two_nodes.sort(key=lambda x: (x[1] << 1) + x[2])

        final_number = (two_nodes[0][1] << 4) + (two_nodes[1][1] << 2)
        first = two_nodes[0][0]
        second = two_nodes[1][0]
        final_number += (int(G.has_edge(first, second)) << 1) + (int(G.has_edge(second, first)) << 4)
        return final_number