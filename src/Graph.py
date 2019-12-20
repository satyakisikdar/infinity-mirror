import networkx as nx


class CustomGraph(nx.Graph):
    """
    Subclass of networkx's Graph class - updates str and repr
    """
    def __init__(self, incoming_graph_data=None, **attr):
        super().__init__(incoming_graph_data=incoming_graph_data, **attr)
        if 'gen_id' in attr:
            self.gen_id = attr['gen_id']

    def __str__(self):
        return f'"{self.name}" gen: {self.gen_id} n:{self.order():_} m:{self.size():_}'

    def __repr__(self):
        return str(self)
