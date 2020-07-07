"""In Out degrees class."""
import numpy as np
import networkx as nx

from ..feature_class import FeatureClass, InterpretabilityScore

featureclass_name = "InOutDegrees"


class InOutDegrees(FeatureClass):
    """In Out degrees class."""

    modes = ["fast", "medium", "slow"]
    shortname = "IOD"
    name = "in_out_degrees"
    encoding = "networkx"

    def compute_features(self):
        
        self.add_feature(
            "in_degree",
            lambda graph: list(dict(graph.in_degree).values()),
            "The distribution of in degrees of each node",
            InterpretabilityScore(3),
            statistics = "centrality",
        )
        
        in_deg_n = lambda graph: [
            i/d for i,d in zip(list(dict(graph.in_degree).values()),list(dict(graph.degree).values()))
        ]
        
        self.add_feature(
            "in_degree_normed",
            in_deg_n,
            "The distribution of the ratio of in and total degrees of each node",
            InterpretabilityScore(3),
            statistics = "centrality",
        )
        
        self.add_feature(
            "out_degree",
            lambda graph: list(dict(graph.out_degree).values()),
            "The distribution of out degrees of each node",
            InterpretabilityScore(3),
            statistics = "centrality",
        )
        
        out_deg_n = lambda graph: [
            o/d for o,d in zip(list(dict(graph.out_degree).values()),list(dict(graph.degree).values()))
        ]
        
        self.add_feature(
            "out_degree_normed",
            out_deg_n,
            "The distribution of the ratio of out and total degrees of each node",
            InterpretabilityScore(3),
            statistics = "centrality",
        )
        
        in_out_deg = lambda graph: [
            i/o for i,o in zip(list(dict(graph.in_degree).values()),list(dict(graph.out_degree).values()))
        ]
        
        self.add_feature(
            "in_out_degree",
            in_out_deg,
            "The distribution of the ratio of in and out degrees of each node",
            InterpretabilityScore(3),
            statistics = "centrality",
        )
        
        self.add_feature(
            "in_degree_centrality",
            lambda graph: list(nx.in_degree_centrality(graph).values()),
            "The distribution of in degree centralities",
            InterpretabilityScore(3),
            statistics = "centrality",
        )
        
        self.add_feature(
            "out_degree_centrality",
            lambda graph: list(nx.out_degree_centrality(graph).values()),
            "The distribution of out degree centralities",
            InterpretabilityScore(3),
            statistics = "centrality",
        )