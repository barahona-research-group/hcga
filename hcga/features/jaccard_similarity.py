"""Jaccard Similarity class."""
from functools import lru_cache
import networkx as nx
import numpy as np

from .utils import ensure_connected,  remove_selfloops
from ..feature_class import FeatureClass, InterpretabilityScore

featureclass_name = "JaccardSimilarity"

"""
Create the Jaccard similarity matrix for nodes in the network.
This is defined as a/(a+b+c), where
a = number of common neighbours
b = number of neighbours of node 1 that are not neighbours of node 2
c = number of neighbours of node 2 that are not neighbours of node 1
Treating this matrix as an adjacency matrix, we can compute network some features
ref: https://www.biorxiv.org/content/10.1101/112540v4.full

For some features we remove selfloops, since the diagonal of the Jaccard 
similarity consists of ones, and therefore all nodes will have a selfloop with weight one
"""

class JaccardSimilarity(FeatureClass):
    """Jaccard Similarity class."""
    
    modes = ["fast", "medium", "slow"]
    shortname = "JS"
    name = "jaccard_similarity"
    encoding = "networkx"

    def compute_features(self):
        
        def jaccard_similarity(graph):
    
            # Construct a graph from Jaccard similarity matrix
            n = nx.number_of_nodes(graph)
            jsm = np.eye(n)
            
            neighbors = [0 for i in range(n)]
            
            for j in range(n):
                neighbors[j] = set(graph.neighbors(j))
                
            for i in range(n):
                for j in range(i+1,n):
                    a = len(neighbors[i].intersection(neighbors[j]))
                    if a == 0:
                        jsm[i,j] = 0
                    else:
                        b = len(neighbors[i].difference(neighbors[j]))
                        c = len(neighbors[j].difference(neighbors[i]))
                        jsm[i,j] = a/(a+b+c)
            
            return nx.Graph(jsm)
        
        g = jaccard_similarity(self.graph)
        
        # Basic stats
        self.add_feature(
            "number_of_edges",
            lambda graph: graph.number_of_edges(),
            "Number of edges in Jaccard similarity graph",
            InterpretabilityScore(5),
            function_args=g,
        )
        
        self.add_feature(
            "number_of_edges_no_selfloops",
            lambda graph: remove_selfloops(graph).number_of_edges(),
            "Number of edges, not including selfloops, in Jaccard similarity graph",
            InterpretabilityScore(5),
            function_args=g,
        )
        
        self.add_feature(
            "connectance",
            lambda graph: nx.density(graph),
            "Connectance of Jaccard similarity graph",
            InterpretabilityScore(5),
            function_args=g,
        )
        
        self.add_feature(
            "diameter",
            lambda graph: nx.diameter(ensure_connected(graph)),
            "Diameter of Jaccard similarity graph",
            InterpretabilityScore(5),
            function_args=g,
        )
        
        self.add_feature(
            "radius",
            lambda graph: nx.radius(ensure_connected(graph)),
            "Radius of Jaccard similarity graph",
            InterpretabilityScore(5),
            function_args=g,
        )
        
        self.add_feature(
            "edge_weights",
            lambda graph: list(nx.get_edge_attributes(remove_selfloops(graph), "weight").values()),
            "Weights of the edges in Jaccard similarity graph",
            InterpretabilityScore(5),
            function_args=g,
            statistics="centrality",
        )
        
        # Assortativity
        self.add_feature(
            "degree_assortativity_coeff",
            lambda graph: nx.degree_assortativity_coefficient(graph),
            "Similarity of connections in Jaccard similarity graph with respect to the node degree",
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
            lambda graph: len(list(nx.connected_components(graph))),
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
        
