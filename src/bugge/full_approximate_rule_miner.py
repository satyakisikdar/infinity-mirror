from networkx.readwrite import sparse6
import networkx as nx
from networkx import utils
from src.bugge.rule_miner_base import *
import random
import sys
from src.bugge.approximate_rule_utils import *
from src.bugge.rule_lib import *
from src.bugge.rule_pq import *
import math

class FullApproximateRuleMiner(RuleMinerBase):
    """Used to find and compress grammar rules in a graph"""

    def __init__(self, G, min_rule_size, max_rule_size, shortcut=None):
        self._G = G
        self.c = min_rule_size
        self.k = max_rule_size
        self.edge_interp = BiDirectionalEdgeTypeInterpreter()
        self.rule_lib = RuleLib()
        self.utils = ApproximateRuleUtils(self.edge_interp, self.rule_lib)
        self.shortcut = shortcut

        self.first_round = True
        self.total_edges_approximated = 0

        self.already_encoded_rules = set()
        self.bits_per_node_id = int(math.ceil(math.log(len(G.nodes), 2)))

        self.cost_to_encode_v = 2 * self.bits_per_node_id - 1

        self.compression_cost = self.bits_per_node_id # Will need a number to say how many rules were used. At most (# nodes - 1) rules will be used.

        self.original_num_nodes = len(G.nodes)
        self.num_nodes = self.original_num_nodes
        self.num_edges = len(G.edges)

        # Removes the extra bits_per_node_id that is necessary when some compression has been done.
        self.original_total_cost = self.cost_to_encode_v + self.cost_of_remaining_edge_list() - self.bits_per_node_id
        self.best_total_cost = self.original_total_cost
        self.best_total_nodes = self.num_nodes
        self.total_cost = self.best_total_cost

        # cost_of_sparse_matrix = len(sparse6.to_sparse6_bytes(G, header=False)) * 8 # An ascii character compression.
        # self.cost_of_original_matrix = cost_of_sparse_matrix # min(cost_of_sparse_matrix, cost_of_dense_matrix)

        self.in_sets = {}
        self.out_sets = {}
        self.neighbors = {}
        for node in list(self._G.nodes()):
            in_set = set([edge[0] for edge in self._G.in_edges(node)])
            out_set = set([edge[1] for edge in self._G.out_edges(node)])
            self.in_sets[node] = in_set
            self.out_sets[node] = out_set
            self.neighbors[node] = in_set | out_set

        # rule_occurrences_by_tuple goes up to self.k layers deep. At the first layer, occurrences is empty
        self.rule_occurrences_by_tuple = {} # {sorted-tuple-of-nodes: {rule_id: full rule details}} 
        self.rule_occurrences_by_id = {}    # {rule-id: set of tuples}
        # This next item means that rules of size s use O(s^2) space. It's really just every tuple the node is a part of.
        self.rule_occurrences_by_node = {n: set() for n in list(G.nodes())}  # {node-id: set of tuples that this node has a rule with}
        self.rule_priority_queue = AugmentedPQ()

        self.csv_format = False
        if self.csv_format:
            # Column headers are rule_id, collapses, edges_approx, total_cost, rule_details
            print("rule_id, collapses, edges_approx, total_cost, rule_details")

    def cost_of_remaining_edge_list(self):
        # Num of non-compressed nodes, one "no-more-edges" bit per node, one "one-more-edge" bit per edge, and one node id per edge.
        return self.bits_per_node_id + self.num_nodes + (self.bits_per_node_id + 1) * self.num_edges

    # A rule here is the following data:
    # (rule_id, cost, nodes_in_rule, nodes_with_external_edges_by_edge_type, deletions_by_edge_type, additions_by_edge_type)
    #
    # where rule_id and cost are ints, nodes_in_rule and nodes_with_external_edges_by_edge_type are lists of nodes,
    # and deletions_by_edge_type and additions_by_edge_type are lists of pairs of nodes,
    #   where the first node in the pair is always the node interior to the rule

    # Assumes that the tuples are sorted.
    def store_rules(self, rules, affected_rule_ids_set):
        t = rules[0][2]

        self.rule_occurrences_by_tuple[t] = {rule[0]: rule for rule in rules}
        for rule in rules:
            rule_id = rule[0]
            cost = rule[1]
            if rule_id not in self.rule_occurrences_by_id:
                self.rule_occurrences_by_id[rule_id] = RulePQ()
            self.rule_occurrences_by_id[rule_id].push(t, cost)
            affected_rule_ids_set.add(rule_id)
        for node in t:
            if node not in self.rule_occurrences_by_node:
                self.rule_occurrences_by_node[node] = set([t])
            else:
                self.rule_occurrences_by_node[node].add(t)

    def delete_node_from_rule_occurrences(self, node_id, rules_affected):
        if node_id not in self.rule_occurrences_by_node:
            return
        tuples = [t for t in self.rule_occurrences_by_node[node_id]]
        for t in tuples:
            # Delete this tuple from rules-by-nodes.
            for node in t:
                if node != node_id:
                    self.rule_occurrences_by_node[node].remove(t)
            # Delete this tuple from rules-by-ids
            for rule_id, rule in self.rule_occurrences_by_tuple[t].items():
                self.rule_occurrences_by_id[rule_id].delete(t)
                if self.rule_occurrences_by_id[rule_id].empty():
                    del self.rule_occurrences_by_id[rule_id]
                rules_affected.add(rule_id)
            # Delete this tuple from rules-by-tuples
            del self.rule_occurrences_by_tuple[t]
        del self.rule_occurrences_by_node[node_id]

    def update_rule_pq(self, rules_affected, rule_currently_using=-1):
        for rule_id in rules_affected:
            if rule_id in self.rule_occurrences_by_id:
                pcd = self.determine_rule_pcd(rule_id, rule_id == rule_currently_using)
                if self.rule_priority_queue.contains(rule_id):
                    self.rule_priority_queue.update(rule_id, pcd)
                else:
                    self.rule_priority_queue.push(rule_id, pcd)
            elif self.rule_priority_queue.contains(rule_id):
                    self.rule_priority_queue.delete(rule_id)

    # This function is intended to be run just once at the start.
    # It looks at every tuple of connected nodes up to size self.k and finds all rules for the respective tuples.
    # The information is stored in self.rule_occurrences_by_tuple and self.rule_occurrences_by_id.
    def update_rules_for_tuples(self, rules_affected, nodes_to_look_at=None):
        always_filter_by_higher_id = False
        if nodes_to_look_at is None:
            nodes_to_look_at = list(self._G.nodes())
            if self.shortcut is not None:
                nodes_to_look_at = list(np.random.permutation(nodes_to_look_at)) # Removes 'adversarial' time until cheapest cost is found
            always_filter_by_higher_id = True
        nodes_to_look_at_set = set(nodes_to_look_at)

        # First, delete any rules involving these nodes.
        for node in nodes_to_look_at:
            self.delete_node_from_rule_occurrences(node, rules_affected)
        
        cheapest_cost = len(self.neighbors) * self.k
        if self.shortcut is not None and len(self.rule_occurrences_by_id) > 0:
            for rule_id, occurrences in self.rule_occurrences_by_id.items():
                t = occurrences.top_item()
                full_rule_details = self.rule_occurrences_by_tuple[t][rule_id]
                if full_rule_details[1] < cheapest_cost:
                    cheapest_cost = full_rule_details[1]
                if cheapest_cost == 0:
                    break
        """
        if self.shortcut is not None and self.rule_priority_queue.size() > 0:
            self.update_rule_pq(rules_affected)
            if self.rule_priority_queue.size() > 0:
                best_rule_id = self.rule_priority_queue.top_item()
                t = self.rule_occurrences_by_id[best_rule_id].top_item()
                full_rule_details = self.rule_occurrences_by_tuple[t][best_rule_id]
                cheapest_cost = full_rule_details[1]
        """

        # Then, add new rules.
        for first_node in nodes_to_look_at:
            # Do a bfs up to depth self.k to give nodes temporary labels.
            # All nodes within h hops of first_node will have ids less than nodes h+1 hops away.
            # We only include nodes > first_node.
            # Also, this part creates a new neighbor set, which only points to neighbors in the next depth.
            # Note that alternate_neighbors uses the old ids.
            alternate_ids = {}
            alternate_neighbors = {}
            seen = set()
            to_explore = [set([first_node])] + [set() for depth in range(1, self.k + 1)]
            next_id = 0
            for depth in range(0, self.k + 1):
                seen |= to_explore[depth]
                for node in to_explore[depth]:
                    alternate_ids[node] = next_id
                    next_id += 1
                    # Only add nodes with higher ids than
                    if depth < self.k:
                        if always_filter_by_higher_id:
                            alternate_neighbors[node] = set([n for n in self.neighbors[node] if n > first_node]) - seen
                        else:
                            alternate_neighbors[node] = set([n for n in self.neighbors[node] if n > first_node or n not in nodes_to_look_at_set]) - seen
                        to_explore[depth + 1] |= alternate_neighbors[node]
                    else:
                        alternate_neighbors[node] = set()

            # Sort the neighbors in reverse order.
            for node in seen:
                alternate_neighbors[node] = [n for n in alternate_neighbors[node]]
                alternate_neighbors[node].sort(key = lambda x: -alternate_ids[x])

            # Now that we have alternate ids and a limited neighbor set, we can traverse the nodes for tuples.
            # This loop maintains the following invariant:
            # 1. (n in frontiers[-1]) --> (forall m in node_stack: alternate_ids[n] > alternate_ids[m])
            # 2. frontiers[-1] is sorted in reverse order by alternate id.
            node_stack = [first_node]
            frontier_stack = [alternate_neighbors[first_node]]
            while len(node_stack) > 0:
                if len(frontier_stack[-1]) == 0:
                    frontier_stack.pop()
                    node_stack.pop()
                    continue
                next_node = frontier_stack[-1].pop()
                node_stack.append(next_node)
                cost = 0
                if len(node_stack) >= self.c:
                    node_stack_copy = [n for n in node_stack]
                    node_stack_copy.sort()
                    rules = self.utils.cheapest_rules_for_tuple([self.out_sets, self.in_sets], tuple(node_stack_copy))
                    cost = rules[0][1]
                    if self.shortcut is None or cost <= cheapest_cost + 1:
                        self.store_rules(rules, rules_affected)
                    if cost < cheapest_cost:
                        cheapest_cost = cost

                continue_expanding = False
                if self.shortcut is None or len(node_stack) == self.k:
                    continue_expanding = len(node_stack) < self.k
                else:
                    allowed_extra_cost = min(self.shortcut + int(math.log(self.k - len(node_stack))), 1 + self.k - len(node_stack))
                    continue_expanding = (cost - cheapest_cost) <= allowed_extra_cost
                if continue_expanding:
                    new_frontier = set(frontier_stack[-1]) | set(alternate_neighbors[next_node])
                    new_frontier = [n for n in new_frontier if alternate_ids[n] > alternate_ids[next_node]]
                    new_frontier.sort(key = lambda x: -alternate_ids[x])
                else:
                    new_frontier = []
                frontier_stack.append(new_frontier)

    # O(1)
    def add_edge(self, source, target):
        self.neighbors[source].add(target)
        self.neighbors[target].add(source)
        self.out_sets[source].add(target)
        self.in_sets[target].add(source)
        self.num_edges += 1

    # O(1)
    def remove_edge(self, source, target):
        if source not in self.out_sets[target]: # If there isn't an edge pointing the other way...
            self.neighbors[source].remove(target)
            self.neighbors[target].remove(source)
        self.out_sets[source].remove(target)
        self.in_sets[target].remove(source)
        self.num_edges -= 1

    # This is O(degree(node_id)) = O(max_degree).
    def delete_node_from_edge_lists(self, node_id):
        for in_neighbor in list(self.in_sets[node_id]): # The typecasting to a list prevents throwing of an error that set is being changed while looping.
            self.remove_edge(in_neighbor, node_id)
        for out_neighbor in list(self.out_sets[node_id]):
            self.remove_edge(node_id, out_neighbor)
        del self.neighbors[node_id]
        del self.in_sets[node_id]
        del self.out_sets[node_id]
        self.num_nodes -= 1

    # A rule here is the following data:
    # (rule_id, cost, nodes_in_rule, nodes_with_external_edges_by_edge_type, deletions_by_edge_type, additions_by_edge_type)
    def collapse_rule(self, rule):
        t = rule[2]
        deletions_by_type = rule[4]
        additions_by_type = rule[5]

        out_node = t[0]
        in_node = t[0]
        if len(rule[3][0]) > 0:
            out_node = rule[3][0].pop()
            rule[3][0].add(out_node)
        if len(rule[3][1]) > 0:
            in_node = rule[3][1].pop()
            rule[3][1].add(in_node)
        out_neighbors = set(self.out_sets[out_node]) - set(t)
        in_neighbors = set(self.in_sets[in_node]) - set(t)

        # Add nodes which have edges being adjusted. Also, adjust the in and out sets based on the representative nodes.
        to_check = set()
        for type_idx in range(0, 2):
            for (a, b) in deletions_by_type[type_idx]:
                to_check.add(b)
                if type_idx == 0 and out_node == a:
                    out_neighbors.remove(b)
                elif type_idx == 1 and in_node == a:
                    in_neighbors.remove(b)
        for type_idx in range(0, 2):
            for (a, b) in additions_by_type[type_idx]:
                to_check.add(b)
                if type_idx == 0 and out_node == a:
                    out_neighbors.add(b)
                elif type_idx == 1 and in_node == a:
                    in_neighbors.add(b)

        # Also add nodes which may have two edges collapsed into 1:
        for i in range(0, len(t)):
            for j in range(i + 1, len(t)):
                node_a = t[i]
                node_b = t[j]
                to_check = to_check | (self.out_sets[node_a] & self.out_sets[node_b]) | (self.in_sets[node_a] & self.in_sets[node_b])
                to_check = to_check | (self.out_sets[node_b] - self.out_sets[node_a]) | (self.in_sets[node_b] - self.in_sets[node_a])

        # Remove nodes in the tuple, except for the single remaining node.
        for i in range(1, len(t)):
            to_check.discard(t[i])
        to_check.add(t[0])

        # Delete nodes all but a single node from edge lists.
        rules_affected = set()
        for i in range(1, len(t)):
            self.delete_node_from_edge_lists(t[i])
            self.delete_node_from_rule_occurrences(t[i], rules_affected)

        # Adjust single remaining node.
        out_additions_for_t0 = out_neighbors - self.out_sets[t[0]]
        out_deletions_for_t0 = self.out_sets[t[0]] - out_neighbors
        in_additions_for_t0 = in_neighbors - self.in_sets[t[0]]
        in_deletions_for_t0 = self.in_sets[t[0]] - in_neighbors
        for neighbor in out_additions_for_t0:
            self.add_edge(t[0], neighbor)
        for neighbor in out_deletions_for_t0:
            self.remove_edge(t[0], neighbor)
        for neighbor in in_additions_for_t0:
            self.add_edge(neighbor, t[0])
        for neighbor in in_deletions_for_t0:
            self.remove_edge(neighbor, t[0])

        self.update_rules_for_tuples(rules_affected, to_check)
        self.update_rule_pq(rules_affected, rule[0])

    # Note that this function makes multiple assumptions:
    # 1. The number of nodes covered by a rule at a given cost will be the number of nodes collapsable by that rule at that cost.
    # 2. The number of rule instances it will take to collapse n nodes is n / size-of-rule.
    # 3. The set of nodes covered by rule instances of cost a is disjoint from the set of nodes covered by rule instances of cost b.
    def determine_rule_pcd(self, rule_id, already_using_rule):
        rule_size = len(self.rule_occurrences_by_id[rule_id].top_item())
        bits_per_rule_node = int(math.ceil(math.log(rule_size, 2)))

        cost_to_encode = 0
        if rule_id not in self.already_encoded_rules:
            cost_to_encode += self.bits_per_node_id # Stores number of nodes in rule.
            cost_to_encode += rule_size * (bits_per_rule_node + 2) # The + 2 is for each node to have an id and a bit about external in/out edges.
            cost_to_encode += rule_size * (rule_size - 1) # Each node also gets bits for whether it points to the other nodes.
            cost_to_encode += 1 # Bit as to whether or not this is the last rule.

        # If we aren't already using the rule, we need to list the number of times that we actually use it.
        cost_to_id_rule = 0
        if not already_using_rule:
            cost_to_id_rule = self.bits_per_node_id # We will use any rule at most (num-of-nodes - 1) times

        best_pcd = -1.0
        sorted_list_of_costs = self.rule_occurrences_by_id[rule_id].sorted_list_of_prios()
        predicted_residue_cost = 0
        predicted_num_nodes = 0
        total_predicted_rules_used = 0
        for cost in sorted_list_of_costs:
            nodes_at_cost = self.rule_occurrences_by_id[rule_id].number_of_nodes_covered_at_priority(cost)
            predicted_num_nodes += nodes_at_cost

            predicted_rules_used = int(math.ceil((0.0 + nodes_at_cost) / rule_size))
            total_predicted_rules_used += predicted_rules_used

            predicted_residue_cost += predicted_rules_used
            if cost > 0:
                # One edge is a pair of interior-exterior nodes and a bit to say which direction the edge points.
                predicted_residue_cost += cost * (bits_per_rule_node + self.bits_per_node_id + 1) * predicted_rules_used
                predicted_residue_cost += cost * predicted_rules_used # An indicator bit for every residue per rule

            predicted_cost_to_say_which_node = total_predicted_rules_used * self.bits_per_node_id
            current_pcd = (cost_to_encode + cost_to_id_rule + \
                            predicted_residue_cost + predicted_cost_to_say_which_node) / (0.0 + predicted_num_nodes)

            if best_pcd == -1.0 or current_pcd < best_pcd:
                best_pcd = current_pcd
                best_cost = cost

        return best_pcd

    # This has a lot of repeat from determine_rule_pcd. Consider better structuring code.
    def record_actual_cost(self, rule_id, already_using_rule, rule_size, cost):
        bits_per_rule_node = int(math.ceil(math.log(rule_size, 2)))

        cost_to_encode = 0
        if rule_id not in self.already_encoded_rules:
            cost_to_encode += self.bits_per_node_id # Stores number of nodes in rule.
            cost_to_encode += rule_size * (bits_per_rule_node + 2) # The + 2 is for each node to have an id and a bit about external in/out edges.
            cost_to_encode += rule_size * (rule_size - 1) # Each node also gets bits for whether it points to the other nodes.
            cost_to_encode += 1 # Bit as to whether or not this is the last rule.

        # If we aren't already using the rule, we need to list the number of times that we actually use it.
        cost_to_id_rule = 0
        if not already_using_rule:
            cost_to_id_rule = self.bits_per_node_id # We will use any rule at most (num-of-nodes - 1) times

        residue_cost = 1 # An indicator bit for when residue is done with.
        if cost > 0:
            # One edge is a pair of interior-exterior nodes and a bit to say which direction the edge points.
            residue_cost += cost * (bits_per_rule_node + self.bits_per_node_id + 1)
            residue_cost += cost # An indicator bit for every residue

        cost_to_say_which_node = self.bits_per_node_id

        return cost_to_encode + cost_to_id_rule + residue_cost + cost_to_say_which_node
        
    def determine_best_rule(self, using_id=-1):
        if self.first_round:
            rules_affected = set()
            self.update_rules_for_tuples(rules_affected)
            self.update_rule_pq(rules_affected)
            self.first_round = False
        if len(self.rule_occurrences_by_id) == 0:
            return -1
        return self.rule_priority_queue.top_item()

    def contract_valid_tuples(self, rule_id):
        old_edges_approx = self.total_edges_approximated
        collapses = 0
        first_time = True
        while self.determine_best_rule(using_id=rule_id) == rule_id:
            t = self.rule_occurrences_by_id[rule_id].top_item()

            full_rule_details = self.rule_occurrences_by_tuple[t][rule_id]

            self.compression_cost += self.record_actual_cost(rule_id, not first_time, len(t), full_rule_details[1])
            if first_time:
                self.already_encoded_rules.add(rule_id)
                first_time = False

            self.collapse_rule(full_rule_details)

            collapses += 1
            self.total_edges_approximated += full_rule_details[1]

        self.update_rule_pq(set([rule_id])) # Updates rule_pq to know that we are no longer currently using this rule id.

        edges_approx = self.total_edges_approximated - old_edges_approx
        
        rule_graph = self.rule_lib.get_rule_graph_by_size_and_id(len(t) + 2, rule_id)
        self.total_cost = self.cost_to_encode_v + self.compression_cost + self.cost_of_remaining_edge_list()
        if self.total_cost < self.best_total_cost:
            self.best_total_cost = self.total_cost
            self.best_total_nodes = self.num_nodes

        if self.csv_format:
            rule_edges_string = str(rule_graph.edges())
            rule_edges_string = rule_edges_string.replace(","," ")
            # Column headers are rule_id, collapses, edges_approx, total_cost, rule_details
            # print("%s, %s, %s, %s, %s" % (rule_id, collapses, edges_approx, self.total_cost, rule_edges_string))
        else:
            # print("Made %s collapses with rule %s, \tincurring a total of %s approximated edges." % (collapses, rule_id, edges_approx))
            # print("The rule's details are: %s" % rule_graph.edges())
            # print("The total cost in bits thus far is %s\n" % self.total_cost)
            pass
        sys.stdout.flush()
        self.draw = False
        if self.draw:
            self.edge_interp.display_rule_graph(rule_graph, "Made %s collapses with this rule. %s edges were approximated." % (collapses, edges_approx))

        return collapses, edges_approx, rule_id, rule_graph

    def done(self):
        if self.first_round:
            return len(self._G.edges()) == 0
        return len(self.rule_occurrences_by_id) == 0

    def cost_comparison(self):
        print("Basic edge list: %s bits" % self.original_total_cost)
        print("Final compression: %s bits (%s percent)" % \
            (self.total_cost, 100.0 * float(self.total_cost) / self.original_total_cost))
        print("Best compression: %s bits (%s percent) occurred with %s of %s nodes compressed (%s percent)" % \
            (self.best_total_cost, 100.0 * float(self.best_total_cost) / self.original_total_cost, \
             self.original_num_nodes - self.best_total_nodes, self.original_num_nodes, \
             (100.0 * float(self.original_num_nodes - self.best_total_nodes) / self.original_num_nodes)))

    def get_remaining_graph(self):
        G = nx.DiGraph()
        for n, neighbors in self.out_sets.items():
            G.add_node(n)
        for n, neighbors in self.out_sets.items():
            for n2 in neighbors:
                G.add_edge(n, n2)
        return G
