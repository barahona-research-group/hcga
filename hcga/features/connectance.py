#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 29 12:35:19 2020

@author: henrypalasciano
"""

"""Connectance class."""
import networkx as nx

from ..feature_class import FeatureClass, InterpretabilityScore

featureclass_name = "connectance"

"Directed networks only"

class Connectance(FeatureClass):
    """Node clique number class.
    """

    modes = ["fast", "medium", "slow"]
    shortname = "Cns"
    name = "connectancd"
    encoding = "networkx"

    def compute_features(self):

        self.add_feature(
            "connectance",
            lambda graph: nx.number_of_edges(graph)/(nx.number_of_nodes(graph)**2),
            "ratio of number of edges to maximum possible number of edges",
            InterpretabilityScore(3)
        )