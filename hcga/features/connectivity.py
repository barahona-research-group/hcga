"""Connectivity class."""
import numpy as np
import networkx as nx

from ..feature_class import FeatureClass, InterpretabilityScore

featureclass_name = "Connectivity"

# Only for directed networks

class Connectivity(FeatureClass):
    """Connectivity class."""

    modes = ["fast", "medium", "slow"]
    shortname = "Con"
    name = "connectivity"
    encoding = "networkx"
    
    def compute_features(self):
    
        self.add_feature(
                "number_strongly_connected_components",
                lambda graph: nx.number_strongly_connected_components(graph),
                "A strongly connected component is a set of nodes in a directed graph such that each node in the set is reachable from any other node in that set",
                InterpretabilityScore(3),
            )
        
        self.add_feature(
                "strongly_connected_component_sizes",
                lambda graph: [len(i) for i in nx.strongly_connected_components(graph)],
                "the distribution of strongly connected component sizes",
                InterpretabilityScore(3),
                statistics="centrality"
            )
        
        self.add_feature(
                "condensation_nodes",
                lambda graph: nx.condensation(graph).number_of_nodes(),
                "number of nodes in the condensation of the graph",
                InterpretabilityScore(3)
            )
        
        self.add_feature(
                "condensation_edges",
                lambda graph: nx.condensation(graph).number_of_edges(),
                "number of edges in the condensation of the graph",
                InterpretabilityScore(3)
            )
        
        self.add_feature(
                "number_weakly_connected_components",
                lambda graph: nx.number_weakly_connected_components(graph),
                "A weakly connected component is a set of nodes in a directed graph such that there exists as edge between each node and at least one other node in the set",
                InterpretabilityScore(3),
            )
        
        self.add_feature(
                "weakly_connected_component_sizes",
                lambda graph: [len(i) for i in nx.weakly_connected_components(graph)],
                "the distribution of weakly connected component sizes",
                InterpretabilityScore(3),
                statistics="centrality"
            )
        
        self.add_feature(
                "number_attracting_components",
                lambda graph: nx.number_attracting_components(graph),
                "An attracting component is a set of nodes in a directed graph such that that once in that set, all other nodes outside that set are not reachable",
                InterpretabilityScore(3),
            )
        
        self.add_feature(
                "attracting_component_sizes",
                lambda graph: [len(i) for i in nx.attracting_components(graph)],
                "the distribution of attracting component sizes",
                InterpretabilityScore(3),
                statistics="centrality"
            )
        
        
        
        
        
        
        
        
        
        
        
                
    
    