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

import networkx as nx
import numpy as np
from networkx.algorithms import centrality

from ..feature_class import FeatureClass, InterpretabilityScore
from . import utils

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
            statistics="centrality",
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
            statistics="centrality",
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
            statistics="centrality",
        )

        # Edge betweenness centrality
        def edge_betweenness_centrality(graph):
            if len(graph.edges) > 0:
                return list(centrality.edge_betweenness_centrality(graph).values())
            else:
                return [np.nan]

        self.add_feature(
            "edge betweenness centrality",
            edge_betweenness_centrality,
            "Betweenness centrality of an edge e is the sum of the fraction of all-pairs shortest paths that pass through e",
            InterpretabilityScore(4),
            statistics="centrality",
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
            statistics="centrality",
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
            statistics="centrality",
        )

        # Second order centrality
        second_order_centrality = lambda graph: list(
            centrality.second_order_centrality(utils.ensure_connected(graph)).values()
        )

        self.add_feature(
            "second order centrality",
            second_order_centrality,
            "The second order centrality of a given node is the standard deviation of the return times to that node of a perpetual random walk on G",
            InterpretabilityScore(4),
            statistics="centrality",
        )

        # Eigenvector centrality
        eigenvector_centrality = lambda graph: list(
            centrality.eigenvector_centrality(
                utils.ensure_connected(graph), max_iter=500
            ).values()
        )
        self.add_feature(
            "eigenvector centrality",
            eigenvector_centrality,
            "Eigenvector centrality computes the centrality for a node based on the centrality of its neighbors",
            InterpretabilityScore(4),
            statistics="centrality",
        )

        # Katz centrality
        katz_centrality = lambda graph: list(
            centrality.katz_centrality(utils.ensure_connected(graph)).values()
        )
        self.add_feature(
            "katz centrality",
            katz_centrality,
            "Generalisation of eigenvector centrality - Katz centrality computes the centrality for a node based on the centrality of its neighbors",
            InterpretabilityScore(4),
            statistics="centrality",
        )

        # Page Rank
        pagerank = lambda graph: list(nx.pagerank(graph).values())
        self.add_feature(
            "pagerank",
            pagerank,
            "The pagerank computes a ranking of the nodes in the graph based on the structure of the incoming links. ",
            InterpretabilityScore(4),
            statistics="centrality",
        )
