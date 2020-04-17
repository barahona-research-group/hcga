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

featureclass_name = "MinimumCuts"


class MinimumCuts(FeatureClass):
    """
    Minimum cuts class
    """

    modes = ["fast", "medium", "slow"]
    shortname = "MiC"
    name = "minimum_cuts"
    keywords = []
    normalize_features = True

    def compute_features(self):
        """Compute the minimum cuts for the network
        

        Notes
        -----
        Calculations using networkx:
            `Networkx_minimum_cuts <https://networkx.github.io/documentation/stable/reference/algorithms/connectivity.html>`_
        
        """
        self.add_feature(
            "min_node_cut_size'",
            lambda graph: len(nx.minimum_node_cut(graph)),
            "Minimum node cut size",
            InterpretabilityScore("max"),
        )

        self.add_feature(
            "min_edge_cut_size'",
            lambda graph: len(nx.minimum_edge_cut(graph)),
            "Minimum edge cut size",
            InterpretabilityScore("max"),
        )
