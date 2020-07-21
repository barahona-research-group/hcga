"""Role-similarity Based Comparison class."""
import networkx as nx
from networkx.algorithms import centrality
import numpy as np

from .utils import ensure_connected, remove_selfloops
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
    """Rbc computation."""
    a = np.where(nx.adj_matrix(graph).toarray() > 0, 1, 0)
    g = nx.DiGraph(a)

    if nx.is_directed_acyclic_graph(g):
        k = nx.dag_longest_path_length(g)
        beta = 0.95

    else:
        lamb = max(np.linalg.eig(a)[0])
        if lamb != 0:
            beta = 0.95 / lamb
        else:
            beta = 0.95
        k = 10
                    
    n = g.number_of_nodes()
    ones = np.ones(n)
    ba = beta * a
    ba_t = np.transpose(ba)

    x = np.zeros([n, k * 2])
    for i in range(1, k + 1):
        x[:, i - 1] = np.dot(np.linalg.matrix_power(ba, i), ones)
        x[:, i + k - 1] = np.dot(np.linalg.matrix_power(ba_t, i), ones)
    x_norm = normalize(x, axis=1)
    y = np.matmul(x_norm, np.transpose(x_norm))

    return nx.Graph(y)

def compute_feats(graph):
    
    feature_list = [graph.number_of_edges(),
                    remove_selfloops(graph).number_of_edges(),
                    nx.density(graph),
                    nx.diameter(ensure_connected(graph)),
                    nx.radius(ensure_connected(graph)),
                    nx.degree_assortativity_coefficient(graph),
                    nx.graph_clique_number(graph),
                    nx.graph_number_of_cliques(graph),
                    nx.transitivity(graph),
                    nx.is_connected(graph) * 1,
                    nx.number_connected_components(graph),
                    ensure_connected(graph).number_of_nodes(),
                    nx.local_efficiency(graph),
                    nx.global_efficiency(graph),
                    nx.node_connectivity(graph),
                    nx.edge_connectivity(graph),
                    ]
                    
    return feature_list


class RolesimilarityBasedComparison(FeatureClass):
    """Role-similarity Based Comparison class."""

    modes = ["fast", "medium", "slow"]
    shortname = "RBC"
    name = "rbc"
    encoding = "networkx"

    def compute_features(self):
        
        feature_names = [
            "number_of_edges",
            "number_of_edges_no_selfloops",
            "connectance",
            "diameter",
            "radius",
            "degree_assortativity_coeff",
            "graph_clique_number",
            "num_max_cliques",
            "transitivity",
            "is_connected",
            "num_connected_components",
            "largest_connected_component",
            "local_efficiency",
            "global_efficiency",
            "node_connectivity",
            "edge_connectivity",
            ]

        self.add_feature(
            feature_names,
            lambda graph: compute_feats(rbc(graph)),
            "Simple features of rbc matrix",
            InterpretabilityScore(5),
            statistics="list",
        )
        
        