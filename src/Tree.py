from typing import Dict

import networkx as nx
from anytree import Node


class LightTreeNode(Node):
    """
    Light Tree Node, just stores a frozen graph in every node
    """
    def __init__(self, name: str, graph: nx.Graph, parent=None, children=None, **kwargs) -> None:
        super(LightTreeNode, self).__init__(name, parent, children, **kwargs)
        self.graph = nx.freeze(graph)
        return

    def __eq__(self, other) -> bool:
        return self.name == other.name

    def __hash__(self):
        return hash(self.name)


class TreeNode(Node):
    """
    Extends the Node class from anytree to store the graph
    """
    def __init__(self, name: str, graph: nx.Graph, stats: Dict[str, float]=None, stats_seq: Dict[str, float]=None, robustness: Dict[str, float]=None, stats_theta: Dict[str, float]=None, parent=None, children=None, **kwargs) -> None:
        super(TreeNode, self).__init__(name, parent, children, **kwargs)
        self.graph: nx.Graph = graph
        self.stats: Dict[str, float] = stats
        self.stats_seq: Dict[str, float] = stats_seq
        self.robustness: Dict[str, float] = robustness
        self.stats_theta: Dict[str, float] = stats_theta
        return

    def __eq__(self, other) -> bool:
        return self.name == other.name

    def __str__(self) -> str:
        return self.name

    def __hash__(self) -> int:
        return hash(self.name)
