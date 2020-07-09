"""Dijkstra shortest paths class."""
import numpy as np
import networkx as nx

from ..feature_class import FeatureClass, InterpretabilityScore

featureclass_name = "DijkstraShortestPaths"

class DijkstraShortestPaths(FeatureClass):
    """Dijkstra shortest paths class."""

    modes = ["fast", "medium", "slow"]
    shortname = "DSP"
    name = "dijkstra_shortest_paths"
    encoding = "networkx"
    
    def compute_features(self):
        
        def shortest_paths(graph):
            s = list(nx.all_pairs_dijkstra_path_length(graph))
            p = []
            for i in range(len(s)):
                p += list(s[i][1].values())
            return [j for j in p if j>0]
        
        self.add_feature(
                "number_shortest_paths",
                lambda graph: len(shortest_paths(graph)),
                "the number of shortest paths (not counting paths of infinite length)",
                InterpretabilityScore(3),
            )
        
        self.add_feature(
                "shortest_paths_length",
                lambda graph: shortest_paths(graph),
                "the number of shortest path lengths (not counting paths of infinite length)",
                InterpretabilityScore(3),
                statistics="centrality",
            )
        
        def eccentricity(graph):
            s = list(nx.all_pairs_dijkstra_path_length(graph))
            p = []
            for i in range(len(s)):
                p.append(max(list(s[i][1].values())))
            return p
        
        self.add_feature(
                "eccentricity",
                lambda graph: eccentricity(graph),
                "the ditribution of eccentricities (not counting paths of infinite length)",
                InterpretabilityScore(3),
                statistics="centrality"
            )
        
        self.add_feature(
                "diameter",
                lambda graph: max(eccentricity(graph)),
                "the diameter of the graph (not counting paths of infinite length)",
                InterpretabilityScore(3),
            )
        
        self.add_feature(
                "radius",
                lambda graph: min(eccentricity(graph)),
                "the radius of the graph (not counting paths of infinite length)",
                InterpretabilityScore(3),
            )
        
        