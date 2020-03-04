# -*- coding: utf-8 -*-
# This file is part of hcga.
#
# Copyright (C) 2019,
# Robert Peach (r.peach13@imperial.ac.uk),
# Alexis Arnaudon (alexis.arnaudon@epfl.ch),
# https://github.com/ImperialCollegeLondon/hcga.git
#
# hcga is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# hcga is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with hcga.  If not, see <http://www.gnu.org/licenses/>.

from .feature_class import FeatureClass
from .feature_class import InterpretabilityScore
import numpy as np
from networkx.algorithms import centrality
import networkx as nx

featureclass_name = "CentralitiesBasic"


class CentralitiesBasic(FeatureClass):
    """Basic stats class"""

    modes = ["fast", "medium", "slow"]
    shortname = "CB"
    name = "centralities_basic"
    keywords = []
    normalize_features = True

    def compute_features(self):
        """
        Compute some standard centrality measures for the network

        Computed statistics    
        -----
        Put here the list of things that are computed, with corresponding names

        """

        # Degree centrality
        degree_centrality = lambda graph: list(
            centrality.degree_centrality(graph).values()
        )
        self.add_feature(
            "degree centrality",
            degree_centrality,
            "The degree centrality distribution",
            InterpretabilityScore(5),
        )

        # Betweenness Centrality
        betweenness_centrality = lambda graph: list(
            centrality.betweenness_centrality(graph).values()
        )
        self.add_feature(
            "betweenness centrality",
            betweenness_centrality,
            "Betweenness centrality of a node v is the sum of the fraction of all-pairs shortest paths that pass through v",
            InterpretabilityScore(5),
        )

        # Closeness centrality
        closeness_centrality = lambda graph: list(
            centrality.closeness_centrality(graph).values()
        )
        self.add_feature(
            "closeness centrality",
            closeness_centrality,
            "Closeness is the reciprocal of the average shortest path distance",
            InterpretabilityScore(5),
        )

        # Edge betweenness centrality
        edge_betweenness_centrality = lambda graph: list(
            centrality.edge_betweenness_centrality(graph).values()
        )
        self.add_feature(
            "edge betweenness centrality",
            edge_betweenness_centrality,
            "Betweenness centrality of an edge e is the sum of the fraction of all-pairs shortest paths that pass through e",
            InterpretabilityScore(4),
        )

        # Harmonic centrality
        harmonic_centrality = lambda graph: list(
            centrality.harmonic_centrality(graph).values()
        )
        self.add_feature(
            "harmonic centrality",
            harmonic_centrality,
            "Harmonic centrality of a node u is the sum of the reciprocal of the shortest path distances from all other nodes to u",
            InterpretabilityScore(4),
        )

        # Subgraph centrality
        subgraph_centrality = lambda graph: list(
            centrality.subgraph_centrality(graph).values()
        )
        self.add_feature(
            "subgraph centrality",
            subgraph_centrality,
            "The subgraph centrality for a node is the sum of weighted closed walks of all lengths starting and ending at that node.",
            InterpretabilityScore(3),
        )

        # Second order centrality
        def second_order_centrality(graph):
            connected_graph = nx.subgraph(
                graph, max(nx.connected_components(graph), key=len)
            )
            return list(centrality.second_order_centrality(connected_graph).values())

        self.add_feature(
            "second order centrality",
            second_order_centrality,
            "The second order centrality of a given node is the standard deviation of the return times to that node of a perpetual random walk on G",
            InterpretabilityScore(4),
        )

        # Eigenvector centrality
        eigenvector_centrality = lambda graph: list(
            centrality.eigenvector_centrality(graph, max_iter=100).values()
        )
        self.add_feature(
            "eigenvector centrality",
            eigenvector_centrality,
            "Eigenvector centrality computes the centrality for a node based on the centrality of its neighbors",
            InterpretabilityScore(4),
        )

        # Katz centrality
        katz_centrality = lambda graph: list(centrality.katz_centrality(graph).values())
        self.add_feature(
            "katz centrality",
            katz_centrality,
            "Generalisation of eigenvector centrality - Katz centrality computes the centrality for a node based on the centrality of its neighbors",
            InterpretabilityScore(4),
        )
