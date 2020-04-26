from typing import Dict

from anytree import Node
import networkx as nx
# from src.Graph import CustomGraph


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

