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

from ..feature_class import FeatureClass, InterpretabilityScore
from . import utils

featureclass_name = "BasicStats"


class BasicStats(FeatureClass):
    """Basic stats class"""

    modes = ["fast", "medium", "slow"]
    shortname = "BS"
    name = "basic_stats"
    keywords = []
    normalize_features = True

    def compute_features(self):
        """
        Compute some basic stats of the network

        Computed statistics
        -----
        Put here the list of things that are computed, with corresponding names

        """

        # basic normalisation parameters
        n_nodes = lambda graph: len(graph)
        n_edges = lambda graph: len(graph.edges)

        # Adding basic node and edge numbers
        self.add_feature(
            "num_nodes",
            n_nodes,
            "Number of nodes in the graph",
            InterpretabilityScore("max"),
        )
        self.add_feature(
            "num_edges",
            n_edges,
            "Number of edges in the graph",
            InterpretabilityScore("max"),
        )

        # Adding diameter stats
        self.add_feature(
            "diameter",
            lambda graph: nx.diameter(utils.ensure_connected(graph)),
            "Diameter of the graph",
            InterpretabilityScore("max"),
        )
        self.add_feature(
            "radius",
            lambda graph: nx.radius(utils.ensure_connected(graph)),
            "Radius of the graph",
            InterpretabilityScore("max"),
        )

        # Degree stats
        density = lambda graph: np.float64(2 * n_edges(graph)) / np.float64(
            n_nodes(graph) * (n_edges(graph) - 1)
        )
        self.add_feature(
            "density", density, "Density of the graph", InterpretabilityScore("max")
        )

        self.add_feature(
            "edge_weights",
            lambda graph: list(nx.get_edge_attributes(graph,'weight').values()),
            "Weights of the edges",
            InterpretabilityScore("max"),
            statistics="centrality",
        )

