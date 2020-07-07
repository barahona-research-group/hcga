"""Basal nodes class."""
import numpy as np
import networkx as nx

from ..feature_class import FeatureClass, InterpretabilityScore

featureclass_name = "BasalNodes"

"Basal nodes are nodes which have in degree equal to zero"

class BasalNodes(FeatureClass):
    """Basal nodes class."""

    modes = ["fast", "medium", "slow"]
    shortname = "BN"
    name = "basal_nodes"
    encoding = "networkx"

    def compute_features(self):
        
        def basal_nodes_func(graph):
            in_degrees = dict(graph.in_degree)
            return [i for i in in_degrees if in_degrees[i] == 0]
        
        self.add_feature(
            "n_basal_nodes",
            lambda graph: len(basal_nodes_func(graph)),
            "The number of basal nodes",
            InterpretabilityScore(3),
        )
            
        self.add_feature(
            "basal_degrees",
            lambda graph: [dict(graph.out_degree)[i] for i in basal_nodes_func(graph)],
            "The distribution of degrees of basal nodes",
            InterpretabilityScore(3),
            statistics = "centrality",
        )
        
        n_basal_edges = lambda graph: sum([dict(graph.out_degree)[i] for i in basal_nodes_func(graph)])
        
        self.add_feature(
            "n_basal_edges",
            n_basal_edges,
            "The total number of edges connected to basal nodes",
            InterpretabilityScore(3),
        )
        
        
        def exp_basal_edge(graph):
            in_degs = list(dict(graph.in_degree).values())
            r = sum([dict(graph.out_degree)[i] for i in basal_nodes_func(graph)])/(graph.number_of_edges())
            return [i*r for i in in_degs]
        
        self.add_feature(
            "exp_basal_edge",
            lambda graph: exp_basal_edge(graph),
            "The distribution of the expected number of in-edges of each node with basal nodes",
            InterpretabilityScore(3),
            statistics = "centrality",
        )