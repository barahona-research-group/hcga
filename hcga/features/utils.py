"""Utils functions for feature classes."""
import networkx as nx


def ensure_connected(func):
    """Ensures that a graph is connected/weakly-connected.

    Acts as a decorator, or wrapp a function to add features:
        ensure_connected(nx.function)
    """

    def _ensure_connected(graph):
        if isinstance(graph, nx.Graph):
            if nx.is_directed(graph):
                if not nx.is_weakly_connected(graph):
                    return func(graph.subgraph(max(nx.weakly_connected_components(graph), key=len)))
            else:
                if not nx.is_connected(graph):
                    return func(graph.subgraph(max(nx.connected_components(graph), key=len)))
            return func(graph)
        raise Exception("ensure_connected is not implemented for this graph type")

    return _ensure_connected


def remove_selfloops(graph):
    """Return a graph without selfloops."""
    if isinstance(graph, nx.Graph):
        graph_noselfloop = graph.copy()
        selfloops = nx.selfloop_edges(graph)
        graph_noselfloop.remove_edges_from(selfloops)
        return graph_noselfloop
    raise Exception("ensure_conneted is not implemented for this graph type")
