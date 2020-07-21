"""Jaccard Similarity class."""
import networkx as nx
from networkx.algorithms import centrality
import numpy as np

from .utils import ensure_connected, remove_selfloops
from ..feature_class import FeatureClass, InterpretabilityScore

featureclass_name = "JaccardSimilarity"

"""
Create the Jaccard similarity matrix for nodes in the network,
then convert this to a graph and extract some features
This is defined as a/(a+b+c), where
a = number of common neighbours
b = number of neighbours of node 1 that are not neighbours of node 2
c = number of neighbours of node 2 that are not neighbours of node 1
Treating this matrix as an adjacency matrix, we can compute network some features
ref: https://www.biorxiv.org/content/10.1101/112540v4.full

For some features we remove selfloops, since the diagonal of the Jaccard
similarity consists of ones, and therefore all nodes will have a selfloop with weight one
"""


def jaccard_similarity(graph):
    """Construct a graph from Jaccard similarity matrix"""

    n = nx.number_of_nodes(graph)
    jsm = np.eye(n)

    neighbors = [0 for i in range(n)]
            
    for i,j in enumerate(graph.nodes()):
        neighbors[i] = set(graph.neighbors(j))
        
    for i in range(n):
        for j in range(i + 1, n):
            a = len(neighbors[i].intersection(neighbors[j]))
            if a == 0:
                jsm[i, j] = 0
            else:
                b = len(neighbors[i].difference(neighbors[j]))
                c = len(neighbors[j].difference(neighbors[i]))

                jsm[i, j] = a / (a + b + c)

    return nx.Graph(jsm)

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


class JaccardSimilarity(FeatureClass):
    """Jaccard Similarity class."""

    modes = ["fast", "medium", "slow"]
    shortname = "JS"
    name = "jaccard_similarity"
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
            lambda graph: compute_feats(jaccard_similarity(graph)),
            "Simple features of Jaccard similarity matrix",
            InterpretabilityScore(5),
            statistics="list",
        )

