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

featureclass_name = "Vitality"


class Vitality(FeatureClass):
    modes = ["medium", "slow"]
    shortname = "V"
    name = "vitality"
    keywords = []
    normalize_features = True

    def compute_features(self):
        """Compute vitality measures.

        
        Notes
        -----

        
        References
        ----------


        """

        # distribution of vitality
        self.add_feature(
            "vitality",
            lambda graph: list(nx.closeness_vitality(graph).values()),
            "The closeness vitality of a node is the change in the sum of distances between all node pairs when excluding that node",
            InterpretabilityScore(3),
            statistics="centrality",
        )
