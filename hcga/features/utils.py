"""Utils functions for feature classes."""
import networkx as nx


def ensure_connected(graph):
    """Ensures that a graph is connected."""
    if isinstance(graph, nx.Graph):
        if not nx.is_connected(graph):
            return graph.subgraph(max(nx.connected_components(graph), key=len))
        return graph
    raise Exception("ensure_conneted is not implemented for this graph type")

def ensure_connected_directed(graph):
    """Ensures that a directed graph is weakly connected."""
    if isinstance(graph, nx.Graph):
        if not nx.is_weakly_connected(graph):
            return graph.subgraph(max(nx.weakly_connected_components(graph), key=len))
        return graph
    raise Exception("ensure_conneted_directed is not implemented for this graph type")

def remove_selfloops(graph):
    """Return a graph without selfloops."""
    if isinstance(graph, nx.Graph):
        graph_noselfloop = graph.copy()
        selfloops = nx.selfloop_edges(graph)
        graph_noselfloop.remove_edges_from(selfloops)
        return graph_noselfloop
    raise Exception("ensure_conneted is not implemented for this graph type")
