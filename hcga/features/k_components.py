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

from functools import lru_cache

from ..feature_class import FeatureClass, InterpretabilityScore

featureclass_name = "KComponents"


class KComponents(FeatureClass):
    """K Components class
    


    Parameters
    ----------



    Notes
    -----


    References
    ----------

    
    """

    modes = ["medium", "slow"]
    shortname = "KC"
    name = "k_components"
    keywords = []
    normalize_features = True

    def compute_features(self):
        """
        Compute the k components of the network

        Computed statistics
        -----
        Put here the list of things that are computed, with corresponding names

        """

        @lru_cache(maxsize=None)
        def eval_kcomponents(graph):
            """this evaluates the main function and cach it for speed up"""
            return nx.k_components(graph)

        self.add_feature(
            "num_connectivity_levels_k",
            lambda graph: len(eval_kcomponents(graph).keys()),
            "The number of connectivity levels k in the input graphs",
            InterpretabilityScore(3),
        )

        self.add_feature(
            "max_num_components",
            lambda graph: max([len(i) for i in eval_kcomponents(graph).values()]),
            "The maximum number of componenets at any value of k",
            InterpretabilityScore(3),
        )

        self.add_feature(
            "size_max_k_component",
            lambda graph: len(
                eval_kcomponents(graph)[len(eval_kcomponents(graph).keys())][0]
            ),
            "The number of nodes of the component corresponding to the largest k",
            InterpretabilityScore(3),
        )

        self.add_feature(
            "size_2_component",
            lambda graph: len(eval_kcomponents(graph)[2][0]),
            "The number of nodes in k=2 component",
            InterpretabilityScore(3),
        )
