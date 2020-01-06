from typing import Dict

from anytree import Node
import networkx as nx
# from src.Graph import CustomGraph


class TreeNode(Node):
    """
    Extends the Node class from anytree to store the graph
    """
    def __init__(self, name: str, stats: Dict[str, float], graph: nx.Graph, parent=None, children=None, **kwargs) -> None:
        super(TreeNode, self).__init__(name, parent, children, **kwargs)
        self.graph: nx.Graph = graph
        self.stats: Dict[str, float] = stats
        return

    def __eq__(self, other) -> bool:
        return self.name == other.name

    def __str__(self) -> str:
        return self.name

    def __hash__(self) -> int:
        return hash(self.name)

