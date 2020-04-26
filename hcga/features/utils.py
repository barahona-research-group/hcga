import networkx as nx


def ensure_connected(graph):
    """Ensures that a graph is connected"""
    if isinstance(graph, nx.Graph):
        if not nx.is_connected(graph):
            return graph.subgraph(max(nx.connected_components(graph), key=len))
        return graph
    else:
        raise Excpetion("ensure_conneted is not implemented for this graph type")


def remove_selfloops(graph):
    """Return a graph without selfloops."""
    if isinstance(graph, nx.Graph):
        graph_noselfloop = graph.copy()
        selfloops = nx.selfloop_edges(graph)
        graph_noselfloop.remove_edges_from(selfloops)
        return graph_noselfloop
    else:
        raise Excpetion("ensure_conneted is not implemented for this graph type")
