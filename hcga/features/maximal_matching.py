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

featureclass_name = "MaximalMatching"


class MaximalMatching(FeatureClass):
    """
    Maximal matching class
    """

    modes = ["fast", "medium", "slow"]
    shortname = "MM"
    name = "MaximalMatching"
    encoding = 'networkx' 

    def compute_features(self):
        """Compute the maximal matching of the network
        
        A matching is a subset of edges such that no edges are connected to
        the same node. It is maximal if the addition of another edge to this
        subset no longer makes it a matching.
        
           
        Notes
        -----
        Maximal matching calculations using networkx:
            `Networkx_maximal_matching <https://networkx.github.io/documentation/stable/reference/algorithms/matching.html>`_


        """

        self.add_feature(
            "maximal_matching",
            lambda graph: len(nx.maximal_matching(graph)),
            "Maximal matching",
            InterpretabilityScore(4),
        )
