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

featureclass_name = "StructuralHoles"


class StructuralHoles(FeatureClass):
    modes = ["medium", "slow"]
    shortname = "SH"
    name = "structural_holes"
    encoding = "networkx"

    def compute_features(self):
        """Compute structural holes measures.

        
        Notes
        -----

        
        References
        ----------


        """

        # distribution of structural holes constraint
        self.add_feature(
            "constraint",
            lambda graph: list(nx.structuralholes.constraint(graph).values()),
            "The constraint is a measure of the extent to which a node v is invested in those nodes that are themselves invested in the neighbors of v",
            InterpretabilityScore(3),
            statistics="centrality",
        )

        self.add_feature(
            "effective_size",
            lambda graph: list(nx.structuralholes.effective_size(graph).values()),
            "The effective size of a node’s ego network is based on the concept of redundancy. A person’s ego network has redundancy to the extent that her contacts are connected to each other as well. ",
            InterpretabilityScore(3),
            statistics="centrality",
        )
