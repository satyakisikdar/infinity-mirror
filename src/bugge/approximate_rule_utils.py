from src.bugge.rule_lib import *
import random
import math
import matplotlib.pyplot as plt
import numpy as np

# An edge type interpreter lets you modify a graph while abstractly speaking of edge types.
# From:
# "This edge connects a,b of type 1"
# To:
# "This edge is a directed edge from b to a"
# or to:
# "This edge is a bidirectional edge between b and a"
# or to:
# "This edge is .... whatever you code it to be interpreted as."
class EdgeTypeInterpreter:
    def __init__(self):
        self.type_names = []

    def add_edge(self, G, edge_type, node_a, node_b):
        pass

    def remove_edge(self, G, edge_type, node_a, node_b):
        pass

    # edges_by_type should be the following:
    #   [{node_0: set-of-node_a's-neighbors-with-edges-of-type-0, node_1: set...}, {node_0: set-of...-type-1, ...}, ...]
    # 
    # t should be a set() of nodes that the rule consists of
    #
    # nodes_with_external_edges_by_type should be the following:
    #   [set-of-nodes-with-external-edges-of-type-0, set-of-...-type-1, ...]
    def make_rule_graph(self, edges_by_type, t, nodes_with_external_edges_by_type):
        G = nx.DiGraph()

        # Create a node with a self-loop for every edge type.
        # Then chain them together with directed edges.
        name = "type_0"
        G.add_node(name)
        G.add_edge(name, name)
        for i in range(1, len(edges_by_type)):
            prev_name = name
            name = "type_%s" % i
            G.add_node(name)
            G.add_edge(name, name)
            G.add_edge(prev_name, name)

        # Now actually add the nodes in the tuple and the internal edges.
        for node_a in t:
            G.add_node(node_a)
            for node_b in t:
                for type_id in range(0, len(edges_by_type)):
                    if node_b in edges_by_type[type_id][node_a]:
                        self.add_edge(G, type_id, node_a, node_b)

        for type_id in range(0, len(edges_by_type)):
            type_node_id = "type_%s" % type_id
            for node in nodes_with_external_edges_by_type[type_id]:
                G.add_edge(type_node_id, node)

        return G

    def a_push_b(self, a, b):
        max_dist = 3.0
        x_diff = b[0] - a[0]
        y_diff = b[1] - a[1]
        if a[0] == b[0] and a[1] == b[1]:
            x_diff = random.uniform(-1.0, 1.0)
            y_diff = random.uniform(-1.0, 1.0)
        dist = math.sqrt(x_diff*x_diff + y_diff*y_diff)
        if dist > max_dist:
            return (0.0, 0.0)
        direction = (x_diff / dist, y_diff / dist)
        target = (direction[0] * max_dist + a[0], direction[1] * max_dist + a[1])
        return self.a_pull_b(target, b)

    def a_pull_b(self, a, b):
        x_diff = a[0] - b[0]
        y_diff = a[1] - b[1]
        dist = math.sqrt(x_diff*x_diff + y_diff*y_diff)
        x_pull = 0.0
        y_pull = 0.0
        if dist != 0.0:
            x_pull = x_diff * dist
            y_pull = y_diff * dist
        return (x_pull, y_pull)

    def dot(self, a, b):
        return a[0] * b[0] + a[1] * b[1]

    def iterate_positions(self, G, positions):
        movement_scalar = 0.01
        nodes = list(G.nodes())
        node_idx = np.random.permutation(len(nodes))
        for i in range(0, len(node_idx)):
            node = nodes[node_idx[i]]
            x_diff = 0.0
            y_diff = 0.0
            for other_node in G.nodes():
                if other_node == node:
                    continue
                if other_node in G.neighbors(node):
                    (x_d, y_d) = self.a_pull_b(positions[other_node], positions[node])
                    x_diff += x_d
                    y_diff += y_d
                (x_d, y_d) = self.a_push_b(positions[other_node], positions[node])
                x_diff += x_d
                y_diff += y_d
            for edge in G.edges():
                if edge[0] == edge[1]:
                    continue
                edge_direction = (positions[edge[1]][0] - positions[edge[0]][0], \
                    positions[edge[1]][1] - positions[edge[0]][1])
                edge_direction_mag = math.sqrt(edge_direction[0]**2 + edge_direction[1]**2)
                edge_direction = (edge_direction[0] / edge_direction_mag, edge_direction[1] / edge_direction_mag)
                start = self.dot(positions[edge[0]], edge_direction)
                stop = self.dot(positions[edge[1]], edge_direction)
                point = self.dot(positions[node], edge_direction)
                if start < point and point < stop:
                    point = self.dot((positions[node][0] - positions[edge[0]][0], positions[node][1] - positions[edge[0]][1]), edge_direction)
                    push_point = (point * edge_direction[0] + positions[edge[0]][0], \
                        point * edge_direction[1] + positions[edge[0]][1])
                    (x_d, y_d) = self.a_push_b(push_point, positions[node])
                    x_diff += x_d
                    y_diff += y_d
            positions[node] = (movement_scalar * x_diff + positions[node][0], \
                movement_scalar * y_diff + positions[node][1])

    def assign_display_positions_to_nodes(self, G):
        positions = {}
        for node in G.nodes():
            positions[node] = (random.uniform(-1.0, 1.0), random.uniform(-1.0, 1.0))

        num_iterations = 1
        for i in range(0, num_iterations):
            self.iterate_positions(G, positions)
        return positions

    def display_rule_graph(self, G, title, labels=None):
        non_type_nodes = list(set(G.nodes()) - set(["type_%s" % i for i in range(0, len(self.type_names))]))
        non_type_nodes.sort()
        if labels is None:
            labels = {non_type_nodes[i]: "" for i in range(0, len(non_type_nodes))}
            for i in range(0, len(self.type_names)):
                labels["type_%s" % i] = "has_%s_edges" % self.type_names[i]

        ignored_edges = set([("type_%s" % i, "type_%s" % i) for i in range(0, len(self.type_names))] + \
                            [("type_%s" % (i-1), "type_%s" % i) for i in range(0, len(self.type_names))])

        edge_list = list(set(G.edges()) - ignored_edges)
        edge_colors = ['black' if type(edge[0]) is int and type(edge[1]) is int else 'r' for edge in edge_list]

        positions = self.assign_display_positions_to_nodes(G)
        nx.draw_networkx(G, nodelist=non_type_nodes, node_color='blue', labels=labels, node_size=100, edgelist=edge_list, edge_color=edge_colors, pos=positions)

        for i in range(0, 4000):
            self.iterate_positions(G, positions)
            if i % 30 == 0:
                nx.draw_networkx(G, nodelist=non_type_nodes, node_color='black', labels=labels, node_size=100, edgelist=edge_list, edge_color=edge_colors, pos=positions)
        plt.title(title)
        plt.draw()
        plt.show()
        nx.draw_networkx(G, nodelist=non_type_nodes, node_color='black', labels=labels, node_size=100, edgelist=edge_list, edge_color=edge_colors, pos=positions)
        plt.title(title)
        plt.draw()
        plt.show()

# 0 = forward edges (out)
# 1 = backward edges (in)
class BiDirectionalEdgeTypeInterpreter(EdgeTypeInterpreter):
    def __init__(self):
        self.type_names = ["out", "in"]

    def add_edge(self, G, edge_type, node_a, node_b):
        if edge_type == 0:
            G.add_edge(node_a, node_b)
        elif edge_type == 1:
            G.add_edge(node_b, node_a)
        else:
            print("MAJOR ERROR! INVALID EDGE TYPE %s" % edge_type)

    def remove_edge(self, G, edge_type, node_a, node_b):
        if edge_type == 0:
            G.remove_edge(node_a, node_b)
        elif edge_type == 1:
            G.remove_edge(node_b, node_a)
        else:
            print("MAJOR ERROR! INVALID EDGE TYPE %s" % edge_type)

    def display_rule_graph(self, G, title):
        G = nx.DiGraph(G)
        # Swap direction of out edges:
        out_edges = list(G.out_edges("type_0"))
        for edge in out_edges:
            if edge[1] == "type_1":
                continue
            G.remove_edge(edge[0], edge[1])
            G.add_edge(edge[1], edge[0])
        
        labels = {node: "" for node in G.nodes()}
        EdgeTypeInterpreter.display_rule_graph(self, G, title, labels)

# 0 = forward edge only (out)
# 1 = backward edge only (in)
# 2 = both directions
class InOutBothEdgeTypeInterpreter(EdgeTypeInterpreter):
    def __init__(self):
        self.type_names = ["out", "in", "both"]

    def add_edge(self, G, edge_type, node_a, node_b):
        if edge_type == 0:
            G.add_edge(node_a, node_b)
        elif edge_type == 1:
            G.add_edge(node_b, node_a)
        elif edge_type == 2:
            G.add_edge(node_a, node_b)
            G.add_edge(node_b, node_a)
        else:
            print("MAJOR ERROR! INVALID EDGE TYPE %s" % edge_type)

    def remove_edge(self, G, edge_type, node_a, node_b):
        if edge_type == 0:
            G.remove_edge(node_a, node_b)
        elif edge_type == 1:
            G.remove_edge(node_b, node_a)
        elif edge_type == 2:
            G.remove_edge(node_a, node_b)
            G.remove_edge(node_b, node_a)
        else:
            print("MAJOR ERROR! INVALID EDGE TYPE %s" % edge_type)

class ApproximateRuleUtils:
    def __init__(self, edge_type_interpreter, rule_lib):
        self.edge_type_interpreter = edge_type_interpreter
        self.rule_lib = rule_lib

    # Right now this is O(max_degree * |t| * 2^|t|)
    def cheapest_rules_for_tuple(self, edge_types, t):
        if len(t) > 32:
            print("EXTREME ERROR. ASKING FOR A RULE OF MORE THAN 32 NODES. CANNOT DO BIT MATH.")
            exit(1)

        num_edge_types = len(edge_types)

        t_set = set(t)
        best_options_found = [[] for i in range(0, num_edge_types)]
        best_cost_found = 0
        total_cost = 0
        for edge_type_idx in range(0, num_edge_types):
            total_cost += best_cost_found

            edge_type = edge_types[edge_type_idx]
            external_neighbors = {node: set() for node in t} # Maps a node to its external neighbors
            internal_neighbors = {} # Maps an external neighbor to its neighbors in t
            max_possible_cost = 0
            for node in t:
                for neighbor in edge_type[node]:
                    if neighbor in t_set:
                        continue
                    external_neighbors[node].add(neighbor)
                    if neighbor not in internal_neighbors:
                        internal_neighbors[neighbor] = set([node])
                    else:
                        internal_neighbors[neighbor].add(node)
                    max_possible_cost += 1

            best_options_found[edge_type_idx] = []
            best_cost_found = max_possible_cost
            for i in range(0, 2**len(t)):
                keep_nodes = set([t[j] for j in range(0, len(t)) if (i >> j) & 1])
                reject_nodes = set([t[j] for j in range(0, len(t)) if not ((i >> j) & 1)])

                # Get the cost for the current option.
                current_cost = 0
                for node in reject_nodes:
                    current_cost += len(external_neighbors[node])
                    if current_cost > best_cost_found:
                        break
                if current_cost > best_cost_found:
                    continue
                # O(|t|*dmax*|t|)
                for neighbor, internals in internal_neighbors.items():
                    matches = internals & keep_nodes
                    current_cost += min(len(matches), len(keep_nodes) - len(matches))
                    if current_cost > best_cost_found:
                        break
                if current_cost > best_cost_found:
                    continue
                if current_cost < best_cost_found:
                    best_options_found[edge_type_idx] = []
                    best_cost_found = current_cost

                # Now that we know the cost is ok, store which edges get added or deleted.
                deletions = []
                additions = []
                for node in reject_nodes:
                    deletions += [(node, neighbor) for neighbor in external_neighbors[node]]
                for neighbor, internals in internal_neighbors.items():
                    matches = internals & keep_nodes
                    non_matches = keep_nodes - matches
                    if len(matches) <= len(non_matches):
                        deletions += [(node, neighbor) for node in matches]
                    else:
                        additions += [(node, neighbor) for node in non_matches]

                use_KT = True
                if not use_KT:
                    # Lastly, check that this isn't making a node a "keep" node when it has no external edges.
                    external_degrees = {node: len(neighbors) for node, neighbors in external_neighbors.items()}
                    for deletion in deletions:
                        external_degrees[deletion[0]] -= 1
                    for addition in additions:
                        external_degrees[addition[0]] += 1
                    invalid = False
                    for node in keep_nodes:
                        if external_degrees[node] == 0:
                            invalid = True
                            break
                    if invalid:
                        continue

                best_options_found[edge_type_idx].append((keep_nodes, deletions, additions))
        total_cost += best_cost_found

        rule_ids = set()
        combined_edge_type_options = []
        sizes = [len(best_options_found[edge_type]) for edge_type in range(0, num_edge_types)]
        counters = [0 for edge_type in range(0, num_edge_types + 1)] # Added a dummy value at the end to make subsequent code simpler.
        counter_idx = 0
        while counter_idx < num_edge_types:
            # Get rule id:
            keep_nodes_by_edge_type = [best_options_found[i][counters[i]][0] for i in range(0, num_edge_types)]
            rule_id = self.rule_lib.add_rule(self.edge_type_interpreter.make_rule_graph(edge_types, t, keep_nodes_by_edge_type))
            # If the rule id is new for this tuple, add it to our set of results.
            if rule_id not in rule_ids:
                rule_ids.add(rule_id)
                deletions_by_edge_type = [best_options_found[i][counters[i]][1] for i in range(0, num_edge_types)]
                additions_by_edge_type = [best_options_found[i][counters[i]][2] for i in range(0, num_edge_types)]
                combined_edge_type_options.append((rule_id, total_cost, t, keep_nodes_by_edge_type, deletions_by_edge_type, additions_by_edge_type))
            
            # Manage the counters
            counter_idx = 0
            counters[counter_idx] += 1
            while counter_idx < num_edge_types and counters[counter_idx] == sizes[counter_idx]:
                counters[counter_idx] = 0
                counter_idx += 1
                counters[counter_idx] += 1

        return combined_edge_type_options
