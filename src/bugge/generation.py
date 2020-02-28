import networkx as nx
from src.bugge.full_approximate_rule_miner import *
import random

# Functions available are:
# naive_fit(G, rule_min, rule_max, shortcut_param=1)
# fit(G, rule_min, rule_max, shortcut_param=1)
# naive_generate(naive_model)
# generate(model)

# Given a __directed__ graph and some BUGGE params, outputs a model of the graph.
def naive_fit(G, rule_min, rule_max, shortcut_param=1):
    assert G.is_directed(), "Attention supreme director; we need some direction over here."
    original_size = len(G.nodes())
    rm = FullApproximateRuleMiner(G, rule_min, rule_max, shortcut_param)

    rules = {}
    occurrences = {}
    edges_approx = {}

    while not rm.done():
        best_rule = rm.determine_best_rule()
        c, ea, ri, rg = rm.contract_valid_tuples(best_rule)
        if ri not in rules:
            rules[ri] = rg
            occurrences[ri] = 0
            edges_approx[ri] = 0
        occurrences[ri] += c
        edges_approx[ri] += ea

    remaining_graph = rm.get_remaining_graph()
    return (original_size, remaining_graph, rules, occurrences, edges_approx)

def fit(G, rule_min, rule_max, shortcut_param=1):
    assert G.is_directed(), "Attention supreme director; we need some direction over here."
    rm = FullApproximateRuleMiner(G, rule_min, rule_max, shortcut_param)

    rules = {}
    steps = []

    while not rm.done():
        best_rule = rm.determine_best_rule()
        c, ea, ri, rg = rm.contract_valid_tuples(best_rule)
        if ri not in rules:
            rules[ri] = rg
        steps.append([c, ea, ri])

    remaining_graph = rm.get_remaining_graph()
    return (remaining_graph, rules, steps)

# Given a model (an output of the "fit" function), generate a __directed__ graph.
def naive_generate(naive_model):
    (original_size, remaining_graph, rules, occurrences, edges_approx) = naive_model
    G = nx.DiGraph(remaining_graph)

    # Prepare to randomly select rules.
    total = 0.0
    for ri, occ in occurrences.items():
        total += occ
    probs = [(ri, float(occ) / total) for ri, occ in occurrences.items()]

    done = False
    while not done:
        # Select a rule stochastically.
        rule = rules[select_from_prob_list(probs)]
        tries = 0
        node = select_node(G)
        # Try a while to find a node the rule can apply to.
        while (not rule_can_apply(G, node, rule)) and tries < len(G.nodes()) * len(G.nodes()):
            node = select_node(G)
            tries += 1
        if rule_can_apply(G, node, rule):
            done = apply_rule_to_node(G, node, rule, original_size)

    G = zero_indexed_graph(G)
    print(sorted(list(G.edges())))
    return G

def generate(model):
    (starting_graph, rules, steps) = model
    G = nx.DiGraph(starting_graph)
    for i in range(0, len(steps)):
        step = steps[len(steps) - (i + 1)]
        count = step[0]
        edges_approx = step[1]
        rule_id = step[2]
        rule = rules[rule_id]
        for j in range(0, count):
            tries = 0
            node = select_node(G)
            # Try a while to find a node the rule can apply to.
            while (not rule_can_apply(G, node, rule)) and tries < len(G.nodes()) * len(G.nodes()):
                node = select_node(G)
                tries += 1
            if rule_can_apply(G, node, rule):
                apply_rule_to_node(G, node, rule, len(G.nodes()) * 1000)
            else:
                print("Generation failed to find a node the rule can apply to! Skipping to next step.")
    G = zero_indexed_graph(G)
    print(sorted(list(G.edges())))
    return G

def zero_indexed_graph(G):
    nodes = sorted(list(G.nodes()))
    old_to_new = {nodes[i]: i for i in range(0, len(nodes))}
    G_new = nx.DiGraph()
    for node in nodes:
        G_new.add_node(old_to_new[node])
    for (source, target) in G.edges():
        G_new.add_edge(old_to_new[source], old_to_new[target])
    return G_new

def rule_can_apply(G, node, rule):
    # 'type_0' is out, 'type_1' is in.
    rule_requires_no_in_edges = len(set(rule.successors('type_1')) - set(['type_1', 'type_0'])) == 0
    rule_requires_no_out_edges = len(set(rule.successors('type_0')) - set(['type_1', 'type_0'])) == 0
    if rule_requires_no_in_edges and len(list(G.predecessors(node))) > 0:
        return False  # Can't apply the rule.
    if rule_requires_no_out_edges and len(list(G.successors(node))) > 0:
        return False  # Can't apply the rule.
    return True

def apply_rule_to_node(G, n, rule, target_size):
    # Then check if the rule would cause us to stop.
    num_new_nodes = len(rule.nodes()) - 3
    diff = (len(G.nodes()) + num_new_nodes) - target_size
    result = diff >= 0
    if result and target_size - len(G.nodes()) < diff:
        return True  # End early if we're closer to the target size than we would be after applying the rule.

    next_node_id = max(G.nodes()) + 1
    input_neighbors = set(G.predecessors(n))
    output_neighbors = set(G.successors(n))

    # Delete the replaced node
    G.remove_node(n)

    # Add the sub-graph
    for r_node in rule.nodes():
        if r_node == 'type_0' or r_node == 'type_1':
            continue
        G.add_node(r_node + next_node_id)
    for r_node in rule.nodes():
        if r_node == 'type_0' or r_node == 'type_1':
            continue
        for out in rule.successors(r_node):
            G.add_edge(r_node + next_node_id, out + next_node_id)
        if ('type_0', r_node) in rule.edges():
            for neighbor in input_neighbors:
                G.add_edge(neighbor, r_node + next_node_id)
        if ('type_1', r_node) in rule.edges():
            for neighbor in output_neighbors:
                G.add_edge(r_node + next_node_id, neighbor)

    return result

def select_from_prob_list(prob_list):
    num = random.random()
    idx = 0
    the_sum = prob_list[0][1]
    while num > the_sum:
        idx += 1
        the_sum += prob_list[idx][1]
    return prob_list[idx][0]

def select_node(G):
    nodes = list(G.nodes())
    idx = random.randint(0, len(nodes) - 1)
    return nodes[idx]
