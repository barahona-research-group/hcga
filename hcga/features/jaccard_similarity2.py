"""Jaccard Similarity class 2."""
import networkx as nx
import numpy as np

from .jaccard_similarity import jaccard_similarity
from .utils import ensure_connected
from ..feature_class import FeatureClass, InterpretabilityScore

featureclass_name = "JaccardSimilarity2"

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


class JaccardSimilarity2(FeatureClass):
    """Jaccard Similarity class2."""

    modes = ["fast", "medium", "slow"]
    shortname = "JS"
    name = "jaccard_similarity2"
    encoding = "networkx"

    def compute_features(self):

        g = jaccard_similarity(self.graph)

        # Cliques
        self.add_feature(
            "graph_clique_number",
            lambda graph: nx.graph_clique_number(graph),
            "The size of the largest clique in the Jaccard similarity graph",
            InterpretabilityScore(3),
            function_args=g,
        )

        self.add_feature(
            "num_max_cliques",
            lambda graph: nx.graph_number_of_cliques(graph),
            "The number of maximal cliques in the Jaccard similarity graph",
            InterpretabilityScore(3),
            function_args=g,
        )

        self.add_feature(
            "num_cliques",
            lambda graph: nx.number_of_cliques(graph),
            "The number of cliques in the Jaccard similarity graph",
            InterpretabilityScore(3),
            function_args=g,
        )

        # Clustering
        self.add_feature(
            "num_triangles",
            lambda graph: np.asarray(list(nx.triangles(graph).values())).mean(),
            "Number of triangles in the Jaccard similarity graph",
            InterpretabilityScore(4),
            function_args=g,
        )

        self.add_feature(
            "transitivity",
            lambda graph: nx.transitivity(graph),
            "Transitivity of the graph",
            InterpretabilityScore(4),
            function_args=g,
        )

        # Components
        self.add_feature(
            "is_connected",
            lambda graph: nx.is_connected(graph) * 1,
            "Whether the Jaccard similarity graph is connected or not",
            InterpretabilityScore(5),
            function_args=g,
        )

        self.add_feature(
            "num_connected_components",
            lambda graph: nx.number_connected_components(graph),
            "The number of connected components",
            InterpretabilityScore(5),
            function_args=g,
        )

        self.add_feature(
            "largest_connected_component",
            lambda graph: ensure_connected(graph).number_of_nodes(),
            "The size of the largest connected component",
            InterpretabilityScore(4),
            function_args=g,
        )

        # Distance measures
        self.add_feature(
            "barycenter_size",
            lambda graph: len(nx.barycenter(ensure_connected(graph))),
            "The barycenter is the subgraph which minimises a distance function",
            InterpretabilityScore(4),
            function_args=g,
        )

        self.add_feature(
            "center_size",
            lambda graph: len(nx.center(ensure_connected(graph))),
            "The center is the subgraph of nodes with eccentricity equal to radius",
            InterpretabilityScore(3),
            function_args=g,
        )

        self.add_feature(
            "center_size",
            lambda graph: nx.extrema_bounding(ensure_connected(graph)),
            "The largest distance in the Jaccard similarity graph",
            InterpretabilityScore(4),
            function_args=g,
        )

        self.add_feature(
            "periphery",
            lambda graph: len(nx.periphery(ensure_connected(graph))),
            "The number of peripheral nodes in the Jaccard similarity graph",
            InterpretabilityScore(4),
            function_args=g,
        )

        self.add_feature(
            "eccentricity",
            lambda graph: list(nx.eccentricity(ensure_connected(graph)).values()),
            "The distribution of node eccentricity across the Jaccard similarity graph",
            InterpretabilityScore(3),
            function_args=g,
            statistics="centrality",
        )

        # Efficiency
        self.add_feature(
            "local_efficiency",
            lambda graph: nx.local_efficiency(graph),
            "The local efficiency",
            InterpretabilityScore(4),
            function_args=g,
        )

        self.add_feature(
            "global_efficiency",
            lambda graph: nx.global_efficiency(graph),
            "The global efficiency",
            InterpretabilityScore(4),
            function_args=g,
        )

        # Independent set
        self.add_feature(
            "size_max_indep_set",
            lambda graph: len(nx.maximal_independent_set(graph)),
            "The number of nodes in the maximal independent set",
            InterpretabilityScore(3),
            function_args=g,
        )

        # Maximal matching
        self.add_feature(
            "maximal_matching",
            lambda graph: len(nx.maximal_matching(graph)),
            "Maximal matching",
            InterpretabilityScore(4),
            function_args=g,
        )

        # Node connectivity
        self.add_feature(
            "node_connectivity",
            lambda graph: nx.node_connectivity(graph),
            "Node connectivity",
            InterpretabilityScore(4),
            function_args=g,
        )

        self.add_feature(
            "avg_node_connectivity",
            lambda graph: nx.average_node_connectivity(graph),
            "Average node connectivity",
            InterpretabilityScore(4),
            function_args=g,
        )

        self.add_feature(
            "edge_connectivity",
            lambda graph: nx.edge_connectivity(graph),
            "Edge connectivity",
            InterpretabilityScore(4),
            function_args=g,
        )

        # Small worldness
        self.add_feature(
            "omega",
            lambda graph: nx.omega(graph),
            "The small world coefficient omega",
            InterpretabilityScore(4),
            function_args=g,
        )
