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


from ..feature_class import FeatureClass, InterpretabilityScore

featureclass_name = "DistanceMeasures"


class DistanceMeasures(FeatureClass):
    """ Distance Measures class """

    modes = ["fast", "medium", "slow"]
    shortname = "DM"
    name = "distance_measures"
    keywords = []
    normalize_features = True

    def compute_features(self):
        """
        Compute the distance measures of the network

        Computed statistics
        -----
        Put here the list of things that are computed, with corresponding names

        """

        # barycenter
        self.add_feature(
            "barycenter_size",
            lambda graph: len(nx.barycenter(graph)),
            "The barycenter is the subgraph which minimises a distance function",
            InterpretabilityScore(4),
        )

        # center
        self.add_feature(
            "center_size",
            lambda graph: len(nx.center(graph)),
            "The center is the subgraph of nodes with eccentricity equal to radius",
            InterpretabilityScore(3),
        )

        # extrema bounding
        self.add_feature(
            "center_size",
            lambda graph: nx.extrema_bounding(graph),
            "The largest distance in the graph",
            InterpretabilityScore(4),
        )

        # periphery
        self.add_feature(
            "periphery",
            lambda graph: len(nx.periphery(graph)),
            "The number of peripheral nodes in the graph",
            InterpretabilityScore(4),
        )

        self.add_feature(
            "eccentricity",
            lambda graph: list(nx.eccentricity(graph).values()),
            "The distribution of node eccentricity across the network",
            InterpretabilityScore(3),
            statistics="centrality",
        )
