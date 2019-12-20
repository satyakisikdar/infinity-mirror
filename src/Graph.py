import networkx as nx
import matplotlib.pyplot as plt


class CustomGraph(nx.Graph):
    """
    Subclass of networkx's Graph class - updates str and repr
    """
    def __init__(self, incoming_graph_data=None, **attr):
        super().__init__(incoming_graph_data=incoming_graph_data, **attr)
        if 'gen_id' in attr:
            self.gen_id = attr['gen_id']
        else:
            self.gen_id = 0

    def __str__(self):
        return f'"{self.name}" gen:{self.gen_id} n:{self.order():_} m:{self.size():_}'

    def __repr__(self):
        return str(self)

    def plot(self, ax=None, prog: str='neato', update_pos: bool=True, pos={}, title=None):
        if update_pos or len(pos) == 0:
            pos = nx.nx_agraph.graphviz_layout(self, prog=prog)

        nx.draw_networkx_nodes(self, pos=pos, alpha=0.5, ax=ax, node_size=20)
        # nx.draw_networkx_labels(self, pos=pos, font_color='w', ax=ax, labels=False)
        nx.draw_networkx_edges(self, pos=pos, ax=ax, alpha=0.15)

        if title is None:
            title = str(self)
        plt.title(title)
        # plt.style.use('seaborn-white')
        plt.grid(False)
