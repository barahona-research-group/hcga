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

featureclass_name = "Eulerian"


class Eulerian(FeatureClass):
    """ Eulerian Measures class """

    modes = ["fast", "medium", "slow"]
    shortname = "EU"
    name = "eulerian"
    keywords = []
    normalize_features = True

    def compute_features(self):
        """
        Compute the Eulerian measures of the network

        Computed statistics
        -----
        Put here the list of things that are computed, with corresponding names

        """

        # checking if eulerian
        self.add_feature(
            "eulerian",
            lambda graph: nx.is_eulerian(graph) * 1,
            "A graph is eulerian if it has a eulerian circuit: a closed walk that includes each edges of the graph exactly once",
            InterpretabilityScore(3),
        )

        # checking if semi eulerian
        self.add_feature(
            "semi_eulerian",
            lambda graph: nx.is_semieulerian(graph) * 1,
            "A graph is semi eulerian if it has a eulerian path but no eulerian circuit",
            InterpretabilityScore(3),
        )

        # checking if eulerian path exists
        self.add_feature(
            "semi_eulerian",
            lambda graph: nx.has_eulerian_path(graph) * 1,
            "Whether a eulerian path exists in the network",
            InterpretabilityScore(3),
        )
