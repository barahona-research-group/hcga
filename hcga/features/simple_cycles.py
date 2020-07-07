"""Simple cycles class."""
import networkx as nx

from ..feature_class import FeatureClass, InterpretabilityScore

featureclass_name = "SimpleCycles"

class SimpleCycles(FeatureClass):
    """Simple cycles class."""

    modes = ["fast", "medium", "slow"]
    shortname = "SC"
    name = "simple_cycles"
    encoding = "networkx"

    def compute_features(self):
        
        def simple_cycles_func(graph):
            return list(nx.simple_cycles(graph))
        
        self.add_feature(
            "number_simple_cycles",
            lambda graph: len(simple_cycles_func(graph)),
            "A simple path with no repeated nodes (except the first) starting and ending at the same node",
            InterpretabilityScore(3)
        )
        
        self.add_feature(
            "simple_cycles_sizes",
            lambda graph: [len(i) for i in simple_cycles_func(graph)],
            "the distribution of simple cycle lengths",
            InterpretabilityScore(3),
            statistics="centrality"
        )