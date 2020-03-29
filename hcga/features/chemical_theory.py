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

from .feature_class import FeatureClass
from .feature_class import InterpretabilityScore
import numpy as np
from networkx.algorithms import centrality
import networkx as nx

featureclass_name = "ChemicalTheory"


class ChemicalTheory(FeatureClass):
    """Chemical theory class"""

    modes = ["fast", "medium", "slow"]
    shortname = "CT"
    name = "chemical_theory"
    keywords = []
    normalize_features = True

    def compute_features(self):
        """
        Compute some standard centrality measures for the network

        Computed statistics    
        -----
        Put here the list of things that are computed, with corresponding names

        """

        wiener_index = lambda graph: [nx.wiener_index(graph)]
        self.add_feature(
            "wiener index",
            wiener_index,
            "The wiener index is defined as the sum of the lengths of the shortest paths between all pairs of vertices",
            InterpretabilityScore(4),
        )

        estrada_index = lambda graph: [nx.estrada_index(graph)]
        self.add_feature(
            "estrada_index",
            estrada_index,
            "The Estrada Index is a topological index of protein folding or 3D compactness",
            InterpretabilityScore(4),
        )
