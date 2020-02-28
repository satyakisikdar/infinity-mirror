import networkx as nx
from networkx import utils
from rule_miner_base import *
from itertools import combinations
from itertools import chain
import random
# from collections import OrderedDict

class ApproximateRuleMiner(RuleMinerBase):
    """Used to find and compress grammar rules in a graph"""

    def __init__(self, G):
        self._G = G
        self.first_round = True
        self.total_edges_approximated = 0

        self.in_sets = {}
        self.out_sets = {}
        self.neighbors = {}
        # self.both_sets = {}
        for node in list(self._G.nodes()):
            in_set = set([edge[0] for edge in self._G.in_edges(node)])
            out_set = set([edge[1] for edge in self._G.out_edges(node)])
            # both_set = in_set | out_set
            # in_only_set = in_set - both_set
            # out_only_set = out_set - both_set
            self.in_sets[node] = in_set # OrderedDict(sorted(in_only_set))
            self.out_sets[node] = out_set # OrderedDict(sorted(out_only_set))
            self.neighbors[node] = in_set | out_set
            # self.both_sets[node] = both_set # OrderedDict(sorted(both_set))

        self.rule_occurrences_by_pair = {}  # {lesser_node_id: {greater_node_id: {rule_id: [adds/deletions]}}}
        self.rule_occurrences_by_id = {}    # {rule_id: Set((lesser_node_id, greater_node_id, cost))}

    def cost_of_an_option(self, option):
        return len((option[0] | option[1]) | (option[2] | option[3])) + \
            len((option[4] | option[5]) | (option[6] | option[7]))

    # This function is intended to be run just once at the start.
    # It looks at every pair of connected nodes and finds all rules for the respective pairs.
    # The information is stored in self.rule_occurrences_by_pair and self.rule_occurrences_by_id.
    def check_all_pairs_for_rules(self):
        nodes = list(self._G.nodes())
        nodes.sort()
        for node_a in nodes:
            self.rule_occurrences_by_pair[node_a] = {}
            for node_b in self.neighbors[node_a]:
                if node_b < node_a:
                    continue
                best_options_without_ids = self.best_options_for_pair(node_a, node_b)
                unique_best_options_with_ids = self.add_rule_ids_and_filter(node_a, node_b, best_options_without_ids)
                self.rule_occurrences_by_pair[node_a][node_b] = unique_best_options_with_ids
                for id_num, option in unique_best_options_with_ids.items():
                    if id_num not in self.rule_occurrences_by_id:
                        self.rule_occurrences_by_id[id_num] = set()
                    self.rule_occurrences_by_id[id_num].add((node_a, node_b, self.cost_of_an_option(option)))

    # This function is run after a rule has contracted some nodes.
    # It updates self.rule_occurrences_by_pair and self.rule_occurrences_by_id.
    # ids contains any node whose rules may have been affected.
    # This is O(|ids| * max_degree * max_degree + |ids|log|ids|) = O(max_degree^3)
    def update_pairs_containing_ids(self, ids):
        ids = list(ids)
        ids.sort()
        for node_c in ids:
            for node_d in self.neighbors[node_c]:
                node_a = min(node_c, node_d)
                node_b = max(node_c, node_d)
                best_options_without_ids = self.best_options_for_pair(node_a, node_b)
                unique_best_options_with_ids = self.add_rule_ids_and_filter(node_a, node_b, best_options_without_ids)
                
                # First delete any outdated occurrences:
                if node_b not in self.rule_occurrences_by_pair[node_a]:
                    self.rule_occurrences_by_pair[node_a][node_b] = {}
                for id_num, option in self.rule_occurrences_by_pair[node_a][node_b].items():
                    if id_num not in unique_best_options_with_ids or \
                            self.cost_of_an_option(unique_best_options_with_ids[id_num]) != self.cost_of_an_option(option):
                        self.rule_occurrences_by_id[id_num].remove((node_a, node_b, self.cost_of_an_option(option)))
                        if len(self.rule_occurrences_by_id[id_num]) == 0:
                            del self.rule_occurrences_by_id[id_num]
                # Then add new occurrences:
                self.rule_occurrences_by_pair[node_a][node_b] = unique_best_options_with_ids
                for id_num, option in unique_best_options_with_ids.items():
                    if id_num not in self.rule_occurrences_by_id:
                        self.rule_occurrences_by_id[id_num] = set()
                    self.rule_occurrences_by_id[id_num].add((node_a, node_b, self.cost_of_an_option(option))) # Adds if not present already.

    # This function is used when a node is deleted from the graph.
    # It deletes all rules containing node_id in self.rule_occurrences_by_*.
    # This is O(degree(node_id)) = O(max_degree).
    def delete_node_from_rule_occurrences(self, node_id):
        node_a = node_id
        for node_b, rules in self.rule_occurrences_by_pair[node_a].items():
            for rule_id, option in rules.items():
                self.rule_occurrences_by_id[rule_id].remove((node_a, node_b, self.cost_of_an_option(option)))
                if len(self.rule_occurrences_by_id[rule_id]) == 0:
                    del self.rule_occurrences_by_id[rule_id]
        del self.rule_occurrences_by_pair[node_a]

        node_1 = node_id
        for node_2 in self.neighbors[node_1]:
            node_a = min(node_1, node_2)
            node_b = max(node_1, node_2)
            if node_a in self.rule_occurrences_by_pair:
                dict_a = self.rule_occurrences_by_pair[node_a]
                if node_b in dict_a:
                    for rule_id, option in dict_a[node_b].items():
                        self.rule_occurrences_by_id[rule_id].remove((node_a, node_b, self.cost_of_an_option(option)))
                        if len(self.rule_occurrences_by_id[rule_id]) == 0:
                            del self.rule_occurrences_by_id[rule_id]
                    del dict_a[node_b]

    def delete_node_pair_from_rule_occurrences(self, node_a, node_b):
        for rule_id, option in self.rule_occurrences_by_pair[node_a][node_b].items():
            self.rule_occurrences_by_id[rule_id].remove((node_a, node_b, self.cost_of_an_option(option)))
            if len(self.rule_occurrences_by_id[rule_id]) == 0:
                del self.rule_occurrences_by_id[rule_id]
        del self.rule_occurrences_by_pair[node_a][node_b]

    # O(1)
    def add_edge(self, source, target):
        self.neighbors[source].add(target)
        self.neighbors[target].add(source)
        self.out_sets[source].add(target)
        self.in_sets[target].add(source)

    # O(1)
    def remove_edge(self, source, target):
        if source not in self.out_sets[target]: # If there isn't an edge pointing the other way...
            self.neighbors[source].remove(target)
            self.neighbors[target].remove(source)
        self.out_sets[source].remove(target)
        self.in_sets[target].remove(source)

    # This is O(degree(node_id)) = O(max_degree).
    def delete_node_from_edge_lists(self, node_id):
        for in_neighbor in list(self.in_sets[node_id]): # The typecasting to a list prevents throwing of an error that set is being changed while looping.
            self.remove_edge(in_neighbor, node_id)
        for out_neighbor in list(self.out_sets[node_id]):
            self.remove_edge(node_id, out_neighbor)
        del self.neighbors[node_id]
        del self.in_sets[node_id]
        del self.out_sets[node_id]

    # This is O(degree(node_a) + degree(node_b)) + O(delete_node_from_edge_lists(node_b)) + O(update_pairs_containing_ids(degree(node_a) + degree(node_b)))
    # Which is O(max_degree^3)
    # But we can provide the tighter bound of O(max_degree + num_edges_changed * max_degree^2)
    def collapse_pair_with_rule(self, node_a, node_b, rule_id):
        # [a_in_add, a_in_del, b_in_add, b_in_del, a_out_add, a_out_del, b_out_add, b_out_del]
        adds_dels = self.rule_occurrences_by_pair[node_a][node_b][rule_id]
        self.delete_node_from_rule_occurrences(node_b) # This one must be called before changing edge lists.

        # Add nodes which have edges being adjusted.
        to_check = ((adds_dels[0] | adds_dels[1]) | (adds_dels[2] | adds_dels[3])) | \
                   ((adds_dels[4] | adds_dels[5]) | (adds_dels[6] | adds_dels[7]))
        self.total_edges_approximated += self.cost_of_an_option(adds_dels)
        # Also add nodes which may have two edges collapsed into 1:
        to_check = to_check | (self.out_sets[node_a] & self.out_sets[node_b]) | (self.in_sets[node_a] & self.in_sets[node_b])
        to_check = to_check | (self.out_sets[node_b] - self.out_sets[node_a]) | (self.in_sets[node_b] - self.in_sets[node_a])

        # Figure out how we're conceptually rewiring things before compressing a and b together.
        new_a_in = (self.in_sets[node_a] - adds_dels[1]) | adds_dels[0]
        new_a_out = (self.out_sets[node_a] - adds_dels[5]) | adds_dels[4]
        new_b_in = (self.in_sets[node_b] - adds_dels[3]) | adds_dels[2]
        new_b_out = (self.out_sets[node_b] - adds_dels[7]) | adds_dels[6]

        # Figure out how to rewire a so that a is equivalent to the compressed version of original a and b.
        actual_a_in_adds = (new_a_in | new_b_in - set([node_a])) - self.in_sets[node_a]
        actual_a_in_dels = self.in_sets[node_a] - (new_a_in | new_b_in - set([node_a]))
        actual_a_out_adds = (new_a_out | new_b_out - set([node_a])) - self.out_sets[node_a]
        actual_a_out_dels = self.out_sets[node_a] - (new_a_out | new_b_out - set([node_a]))

        # Delete node b.
        self.delete_node_from_edge_lists(node_b)

        # Adjust node a.
        for a_in_add in actual_a_in_adds:
            self.add_edge(a_in_add, node_a)
        for a_in_del in actual_a_in_dels:
            self.remove_edge(a_in_del, node_a)
            if a_in_del != node_b and a_in_del not in self.out_sets[node_a]:
                self.delete_node_pair_from_rule_occurrences(min(a_in_del, node_a), max(a_in_del, node_a))
        for a_out_add in actual_a_out_adds:
            self.add_edge(node_a, a_out_add)
        for a_out_del in actual_a_out_dels:
            self.remove_edge(node_a, a_out_del)
            if a_out_del != node_b and a_out_del not in self.in_sets[node_a]:
                self.delete_node_pair_from_rule_occurrences(min(a_out_del, node_a), max(a_out_del, node_a))

        to_check.add(node_a)
        to_check.discard(node_b) # Because b might have had a self-loop that was conceptually deleted, but we don't want to check it.
        self.update_pairs_containing_ids(to_check)

    # O(|V|*max_degree^2) on first run.
    # O(num distinct rule _occurrences_) afterwards.
    def determine_best_rule(self):
        if self.first_round:
            self.check_all_pairs_for_rules()
            self.first_round = False
        best_occ_len = 0
        best_occ_cost = -1
        for id_num, occurrences in self.rule_occurrences_by_id.items():
            if best_occ_cost == -1:
                best_occ_cost = min([t[2] for t in occurrences])
            curr_cost = min([t[2] for t in occurrences])
            if curr_cost < best_occ_cost:
                best_occ_cost = curr_cost
                best_id = id_num
                best_occ_len = len([t for t in occurrences if t[2] == curr_cost])
            elif curr_cost == best_occ_cost:
                length = len([t for t in occurrences if t[2] == curr_cost])
                if length > best_occ_len:
                    best_occ_len = length
                    best_id = id_num
        return [best_id, best_occ_cost]

    def contract_valid_tuples(self, rule_id_with_projected_occurrences):
        rule_id = rule_id_with_projected_occurrences[0]
        cost = rule_id_with_projected_occurrences[1]
        old_edges_approx = self.total_edges_approximated
        collapses = 0
        while rule_id in self.rule_occurrences_by_id and self.determine_best_rule()[1] == cost:
            our_copy = [t for t in self.rule_occurrences_by_id[rule_id]]
            i = 0
            while i < len(our_copy) and our_copy[i][2] > cost:
                i += 1
            if i == len(our_copy):
                break
            (node_a, node_b, cost) = our_copy[i]
            # print("Contracting %s and %s with rule_id %s." % (node_a, node_b, rule_id))
            self.collapse_pair_with_rule(node_a, node_b, rule_id)
            collapses += 1
        edges_approx = self.total_edges_approximated - old_edges_approx
        print("Made %s collapses with rule %s, incurring a total of %s approximated edges." % (collapses, rule_id, edges_approx))


    def done(self):
        for node, neighbors in self.neighbors.items():
            if len(neighbors) > 0:
                return False
        return True

    def random_subset(self, entities):
        subset = []
        for entity in entities:
            if random.randint(0,1):
                subset.append[entity]
        return subset

    # Returns all valid, cheapest ways to edit the pair.
    # Or, if the pair is already valid, returns an empty array.
    # Note that this may currently return duplicates.
    def best_options_for_pair(self, a, b):
        just_a = set([a])
        just_b = set([b])

        in_sets = [self.in_sets[a] - just_b, self.in_sets[b] - just_a]
        in_sets.append(in_sets[0] & in_sets[1])
        out_sets = [self.out_sets[a] - just_b, self.out_sets[b] - just_a]
        out_sets.append(out_sets[0] & out_sets[1])

        three_in_values = [len(in_sets[0]), len(in_sets[1]), len(in_sets[0]) + len(in_sets[1]) - 2 * len(in_sets[2])]
        three_out_values = [len(out_sets[0]), len(out_sets[1]), len(out_sets[0]) + len(out_sets[1]) - 2 * len(out_sets[2])]

        in_min = min(three_in_values)
        out_min = min(three_out_values)

        if in_min == 0 and out_min == 0:
            # Already valid! No modifications needed.
            return [[set() for i in range(0,8)]]

        return_values = []

        # Distinct possible best edit options:
        if three_in_values[0] == in_min:
            if three_out_values[0] == out_min:
                # Delete a_in and delete a_out
                a_in_add = set()
                a_in_del = in_sets[0]
                b_in_add = set()
                b_in_del = set()

                a_out_add = set()
                a_out_del = out_sets[0]
                b_out_add = set()
                b_out_del = set()
                return_values.append([a_in_add, a_in_del, b_in_add, b_in_del, a_out_add, a_out_del, b_out_add, b_out_del])
            if three_out_values[1] == out_min:
                # Delete a_in and delete b_out
                a_in_add = set()
                a_in_del = in_sets[0]
                b_in_add = set()
                b_in_del = set()

                a_out_add = set()
                a_out_del = set()
                b_out_add = set()
                b_out_del = out_sets[1]
                return_values.append([a_in_add, a_in_del, b_in_add, b_in_del, a_out_add, a_out_del, b_out_add, b_out_del])
            if three_out_values[2] == out_min:
                # Delete a_in and move outs to intersection
                # There are actually 2^(out_sets[2]) ways to do this!

                a_only = out_sets[2] - out_sets[0]
                b_only = out_sets[2] - out_sets[1]
                a_only_subset = set(self.random_subset(a_only)) # a will delete this
                b_only_subset = set(self.random_subset(b_only)) # b will delete this

                a_in_add = set()
                a_in_del = in_sets[0]
                b_in_add = set()
                b_in_del = set()

                a_out_add = b_only - b_only_subset # a adds what b does not delete
                a_out_del = a_only_subset
                b_out_add = a_only - a_only_subset # b adds what a does not delete
                b_out_del = b_only_subset
                return_values.append([a_in_add, a_in_del, b_in_add, b_in_del, a_out_add, a_out_del, b_out_add, b_out_del])
        if three_in_values[1] == in_min:
            if three_out_values[0] == out_min:
                # Delete b_in and delete a_out
                a_in_add = set()
                a_in_del = set()
                b_in_add = set()
                b_in_del = in_sets[1]

                a_out_add = set()
                a_out_del = in_sets[0]
                b_out_add = set()
                b_out_del = set()
                return_values.append([a_in_add, a_in_del, b_in_add, b_in_del, a_out_add, a_out_del, b_out_add, b_out_del])
            if three_out_values[1] == out_min:
                # Delete b_in and delete b_out
                a_in_add = set()
                a_in_del = set()
                b_in_add = set()
                b_in_del = in_sets[1]

                a_out_add = set()
                a_out_del = set()
                b_out_add = set()
                b_out_del = out_sets[1]
                return_values.append([a_in_add, a_in_del, b_in_add, b_in_del, a_out_add, a_out_del, b_out_add, b_out_del])
            if three_out_values[2] == out_min:
                # Delete b_in and move outs to intersection
                # There are actually 2^(out_sets[2]) ways to do this!

                a_only = out_sets[2] - out_sets[0]
                b_only = out_sets[2] - out_sets[1]
                a_only_subset = set(self.random_subset(a_only)) # a will delete this
                b_only_subset = set(self.random_subset(b_only)) # b will delete this

                a_in_add = set()
                a_in_del = set()
                b_in_add = set()
                b_in_del = in_sets[1]

                a_out_add = b_only - b_only_subset # a adds what b does not delete
                a_out_del = a_only_subset
                b_out_add = a_only - a_only_subset # b adds what a does not delete
                b_out_del = b_only_subset
                return_values.append([a_in_add, a_in_del, b_in_add, b_in_del, a_out_add, a_out_del, b_out_add, b_out_del])
        if three_in_values[2] == in_min:
            if three_out_values[0] == out_min:
                # Move ins to intersection and delete a_out
                # There are actually 2^(in_sets[2]) ways to do this!

                a_only = in_sets[2] - in_sets[0]
                b_only = in_sets[2] - in_sets[1]
                a_only_subset = set(self.random_subset(a_only)) # a will delete this
                b_only_subset = set(self.random_subset(b_only)) # b will delete this

                a_in_add = b_only - b_only_subset # a adds what b does not delete
                a_in_del = a_only_subset
                b_in_add = a_only - a_only_subset # b adds what a does not delete
                b_in_del = b_only_subset
                
                a_out_add = set()
                a_out_del = out_sets[0]
                b_out_add = set()
                b_out_del = set()
                return_values.append([a_in_add, a_in_del, b_in_add, b_in_del, a_out_add, a_out_del, b_out_add, b_out_del])
            if three_out_values[1] == out_min:
                # Move ins to intersection and delete b_out
                # There are actually 2^(in_sets[2]) ways to do this!

                a_only = in_sets[2] - in_sets[0]
                b_only = in_sets[2] - in_sets[1]
                a_only_subset = set(self.random_subset(a_only)) # a will delete this
                b_only_subset = set(self.random_subset(b_only)) # b will delete this

                a_in_add = b_only - b_only_subset # a adds what b does not delete
                a_in_del = a_only_subset
                b_in_add = a_only - a_only_subset # b adds what a does not delete
                b_in_del = b_only_subset

                a_out_add = set()
                a_out_del = set()
                b_out_add = set()
                b_out_del = out_sets[1]
                return_values.append([a_in_add, a_in_del, b_in_add, b_in_del, a_out_add, a_out_del, b_out_add, b_out_del])
            if three_out_values[2] == out_min:
                # Move both ins and outs to their respective intersections
                # There are actually 2^(in_sets[2] + out_sets[2]) ways to do this!

                a_only = in_sets[2] - in_sets[0]
                b_only = in_sets[2] - in_sets[1]
                a_only_subset = set(self.random_subset(a_only)) # a will delete this
                b_only_subset = set(self.random_subset(b_only)) # b will delete this

                a_in_add = b_only - b_only_subset # a adds what b does not delete
                a_in_del = a_only_subset
                b_in_add = a_only - a_only_subset # b adds what a does not delete
                b_in_del = b_only_subset

                a_only = out_sets[2] - out_sets[0]
                b_only = out_sets[2] - out_sets[1]
                a_only_subset = set(self.random_subset(a_only)) # a will delete this
                b_only_subset = set(self.random_subset(b_only)) # b will delete this

                a_out_add = b_only - b_only_subset # a adds what b does not delete
                a_out_del = a_only_subset
                b_out_add = a_only - a_only_subset # b adds what a does not delete
                b_out_del = b_only_subset
                return_values.append([a_in_add, a_in_del, b_in_add, b_in_del, a_out_add, a_out_del, b_out_add, b_out_del])

        return return_values

    # Adds rule id information AND filters out duplicates
    # Id value will be 6 bits:
    # Bit 5: Does node x have in-edges?
    # Bit 4: Does node x have out-edges?
    # Bit 3: Does node x point to node y?
    # Bit 2: Does node y have in-edges?
    # Bit 1: Does node y have out-edges?
    # Bit 0: Does node y point to node x?
    def add_rule_ids_and_filter(self, a, b, best_options):
        just_a = set([a])
        just_b = set([b])

        in_sets = [self.in_sets[a] - just_b, self.in_sets[b] - just_a]
        out_sets = [self.out_sets[a] - just_b, self.out_sets[b] - just_a]
        
        a_to_b = a in self.in_sets[b]
        b_to_a = b in self.in_sets[a]

        best_options_with_ids = {}
        for option in best_options:
            # [a_in_add = 0, a_in_del = 1, b_in_add = 2, b_in_del = 3, a_out_add = 4, a_out_del = 5, b_out_add = 6, b_out_del = 7]
            in_a = len((in_sets[0] - option[1]) | option[0]) > 0
            in_b = len((in_sets[1] - option[3]) | option[2]) > 0
            out_a = len((out_sets[0] - option[5]) | option[4]) > 0
            out_b = len((out_sets[1] - option[7]) | option[6]) > 0
            a_score = in_a * 4 + out_a * 2 + a_to_b
            b_score = in_b * 4 + out_b * 2 + b_to_a
            if a_score > b_score:
                best_options_with_ids[(a_score << 3) + b_score] = option
            else:
                best_options_with_ids[(b_score << 3) + a_score] = option
        return best_options_with_ids
