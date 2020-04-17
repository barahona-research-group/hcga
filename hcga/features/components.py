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

from functools import lru_cache

import networkx as nx

from ..feature_class import FeatureClass, InterpretabilityScore

featureclass_name = "Components"


class Components(FeatureClass):
    """Components class"""

    modes = ["fast", "medium", "slow"]
    shortname = "C"
    name = "components"
    keywords = []
    normalize_features = True

    def compute_features(self):
        """
        Compute some measured regarding the number of components in the network
        This only computes for directed graphs.

        Computed statistics
        -----
        Put here the list of things that are computed, with corresponding names

        Notes
        -----
        Components calculations using networkx:
            `Networkx_components <https://networkx.github.io/documentation/stable/reference/algorithms/component.html>`_
        
        """

        @lru_cache(maxsize=None)
        def eval_connectedcomponents(graph):
            """this evaluates the main function and cach it for speed up"""
            return list(nx.connected_components(graph))

        self.add_feature(
            "largest_connected_component",
            lambda graph: len(eval_connectedcomponents(graph)[0]),
            "The size of the largest connected component",
            InterpretabilityScore(4),
        )

        def ratio_largest(graph):
            if len(eval_connectedcomponents(graph)) == 1:
                return 0
            return (
                len(eval_connectedcomponents(graph)[0])
                / len(eval_connectedcomponents(graph)[1]),
            )

        self.add_feature(
            "ratio_largest_connected_components",
            ratio_largest,
            "The size ratio of the two largest connected components",
            InterpretabilityScore(4),
        )

        def ratio_min_max(graph):
            if len(eval_connectedcomponents(graph)) == 1:
                return 0
            return (
                len(eval_connectedcomponents(graph)[0])
                / len(eval_connectedcomponents(graph)[-1]),
            )

        self.add_feature(
            "ratio_maxmin_connected_components",
            ratio_min_max,
            "The size ratio of the max and min largest connected components",
            InterpretabilityScore(4),
        )
