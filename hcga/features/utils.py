import networkx as nx


def ensure_connected(graph):
    """Ensures that a graph is connected"""
    if not nx.is_connected(graph):
        return graph.subgraph(max(nx.connected_components(graph), key=len))
    return graph
