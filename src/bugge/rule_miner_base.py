import networkx as nx
from networkx import utils

class RuleMinerBase:
    """Used to find and compress grammar rules in a graph"""

    def __init__(self, G):
        self._G = G

    def determine_best_rule(self):
        pass

    def contract_valid_tuples(self, rule_with_occurrences):
        pass

    def done(self):
        pass