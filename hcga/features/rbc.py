"""Role-similarity Based Comparison class."""
import networkx as nx
import numpy as np
from sklearn.preprocessing import normalize

from .utils import ensure_connected,  remove_selfloops
from ..feature_class import FeatureClass, InterpretabilityScore

featureclass_name = "RolesimilarityBasedComparison"

"""
Create the role-similarity based comparison (rbc) matrix for nodes in the network,
then convert this to a graph and extract some features
ref: https://arxiv.org/abs/1103.5582
For some features we remove selfloops, since the diagonal of the rbc matrix
consists of ones, and therefore all nodes will have a selfloop with weight one
"""

def rbc(graph):
            
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


class RolesimilarityBasedComparison(FeatureClass):
    """Role-similarity Based Comparison class."""
    
    modes = ["fast", "medium", "slow"]
    shortname = "RBC"
    name = "rbc"
    encoding = "networkx"

    def compute_features(self):
        
        g = rbc(self.graph)
        
        # Basic stats
        self.add_feature(
            "number_of_edges",
            lambda graph: graph.number_of_edges(),
            "Number of edges in rbc graph",
            InterpretabilityScore(5),
            function_args=g,
        )
        
        self.add_feature(
            "number_of_edges_no_selfloops",
            lambda graph: remove_selfloops(graph).number_of_edges(),
            "Number of edges, not including selfloops, in rbc graph",
            InterpretabilityScore(5),
            function_args=g,
        )
        
        self.add_feature(
            "connectance",
            lambda graph: nx.density(graph),
            "Connectance of rbc graph",
            InterpretabilityScore(5),
            function_args=g,
        )
        
        self.add_feature(
            "diameter",
            lambda graph: nx.diameter(ensure_connected(graph)),
            "Diameter of rbc graph",
            InterpretabilityScore(5),
            function_args=g,
        )
        
        self.add_feature(
            "radius",
            lambda graph: nx.radius(ensure_connected(graph)),
            "Radius of rbc graph",
            InterpretabilityScore(5),
            function_args=g,
        )
        
        self.add_feature(
            "edge_weights",
            lambda graph: list(nx.get_edge_attributes(remove_selfloops(graph), "weight").values()),
            "Weights of the edges in rbc graph",
            InterpretabilityScore(5),
            function_args=g,
            statistics="centrality",
        )
        
        # Assortativity
        self.add_feature(
            "degree_assortativity_coeff",
            lambda graph: nx.degree_assortativity_coefficient(graph),
            "Similarity of connections in rbc graph with respect to the node degree",
            InterpretabilityScore(4),
            function_args=g,
        )
        
        # Wiener index
        self.add_feature(
            "wiener index",
            lambda graph: nx.wiener_index(graph),
            "The wiener index is defined as the sum of the lengths of the shortest paths between all pairs of vertices",
            InterpretabilityScore(4),
            function_args=g,
        )
        
        # Centralities
        self.add_feature(
            "degree centrality",
            lambda graph: list(nx.degree_centrality(graph).values()),
            "The degree centrality distribution",
            InterpretabilityScore(5),
            function_args=g,
            statistics="centrality",
        )
        
        self.add_feature(
            "eigenvector centrality",
            lambda graph: list(centrality.eigenvector_centrality_numpy(utils.ensure_connected(graph)).values()),
            "Eigenvector centrality computes the centrality for a node based \
            on the centrality of its neighbors",
            InterpretabilityScore(4),
            function_args=g,
            statistics="centrality",
        )
        
        