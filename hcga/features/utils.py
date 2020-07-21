"""Utils functions for feature classes."""
import networkx as nx
import numpy as np
from sklearn.preprocessing import normalize


def ensure_connected(graph):
    """Ensures that a graph is connected/weakly-connected."""
    if isinstance(graph, nx.Graph):
        if nx.is_directed(graph):
            if not nx.is_weakly_connected(graph):
                return graph.subgraph(max(nx.weakly_connected_components(graph), key=len))
        else:
            if not nx.is_connected(graph):
                return graph.subgraph(max(nx.connected_components(graph), key=len))
        return graph
    raise Exception("ensure_connected is not implemented for this graph type")

def remove_selfloops(graph):
    """Return a graph without selfloops."""
    if isinstance(graph, nx.Graph):
        graph_noselfloop = graph.copy()
        selfloops = nx.selfloop_edges(graph)
        graph_noselfloop.remove_edges_from(selfloops)
        return graph_noselfloop
    raise Exception("ensure_conneted is not implemented for this graph type")        
    


def rbc(graph):
    """Construct a graph from role-similarity based comparison matrix"""
    a = np.where(nx.adj_matrix(graph).toarray() > 0, 1, 0)
    g = nx.DiGraph(a)
                    
    if nx.is_directed_acyclic_graph(g):
        k = nx.dag_longest_path_length(g)
        beta = 0.95
                    
    else:
        l = max(np.linalg.eig(a)[0])
        if l != 0:
            beta = 0.95/l
        else:
            beta = 0.95
        k = 10
                    
    n = g.number_of_nodes()
    ones = np.ones(n)
    ba = beta * a
    ba_t = np.transpose(ba)
    x = np.zeros([n, k*2])
    for i in range(1,k+1):
        x[:,i-1] = np.dot(np.linalg.matrix_power(ba,i),ones)
        x[:,i+k-1] = np.dot(np.linalg.matrix_power(ba_t,i),ones)
        
    x_norm = normalize(x, axis=1)
    y = np.matmul(x_norm,np.transpose(x_norm))
                    
    return nx.Graph(y)
