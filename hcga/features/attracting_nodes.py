"""Attracting nodes class."""
import numpy as np
import networkx as nx

from ..feature_class import FeatureClass, InterpretabilityScore

featureclass_name = "AttractingNodes"

"Attracting nodes are nodes which have out degree equal to zero"

class AttractingNodes(FeatureClass):
    """Attracting nodes class."""

    modes = ["fast", "medium", "slow"]
    shortname = "AN"
    name = "attracting_nodes"
    encoding = "networkx"

    def compute_features(self):
        
        def attracting_nodes_func(graph):
            out_degrees = dict(graph.out_degree)
            return [i for i in out_degrees if out_degrees[i] == 0]
        
        self.add_feature(
            "n_attracting_nodes",
            lambda graph: len(attracting_nodes_func(graph)),
            "The number of basal nodes",
            InterpretabilityScore(3),
        )
            
        self.add_feature(
            "attracting_degrees",
            lambda graph: [dict(graph.in_degree)[i] for i in attracting_nodes_func(graph)],
            "The distribution of degrees of attracting nodes",
            InterpretabilityScore(3),
            statistics = "centrality",
        )
        
        n_attracting_edges = lambda graph: sum([dict(graph.in_degree)[i] for i in attracting_nodes_func(graph)])
        
        self.add_feature(
            "n_attracting_edges",
            n_attracting_edges,
            "The total number of edges connected to attracting nodes",
            InterpretabilityScore(3),
        )
        
        def exp_attracting_edge(graph):
            out_degs = list(dict(graph.out_degree).values())
            r = sum([dict(graph.in_degree)[i] for i in attracting_nodes_func(graph)])/(graph.number_of_edges())
            return [i*r for i in out_degs]
        
        self.add_feature(
            "exp_attracting_edge",
            lambda graph: exp_attracting_edge(graph),
            "The distribution of the expected number of out-edges of each node with attracting nodes",
            InterpretabilityScore(3),
            statistics = "centrality",
        )
        