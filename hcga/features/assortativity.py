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
import networkx as nx
from networkx.algorithms import assortativity


featureclass_name = "Assortativity"


class Assortativity(FeatureClass):
    """Basic stats class"""

    modes = ["fast", "medium", "slow"]
    shortname = "AS"
    name = "assortativity"
    keywords = []
    normalize_features = True

    def compute_features(self):
        """
        Compute the assortativity of the network structure

        Computed statistics    
        -----
        Put here the list of things that are computed, with corresponding names

        """

        # Adding basic node and edge numbers
        self.add_feature(
            "degree_assortativity_coeff",
            lambda graph: nx.degree_assortativity_coefficient(graph),
            "Similarity of connections in the graph with respect to the node degree",
            InterpretabilityScore(4),
        )

        self.add_feature(
            "degree_assortativity_coeff_pearson",
            lambda graph: nx.degree_pearson_correlation_coefficient(graph),
            "Similarity of connections in the graph with respect to the node degree",
            InterpretabilityScore(4),
        )

        average_neighbor_degree = lambda graph: np.asarray(
            list(assortativity.average_neighbor_degree(graph).values())
        )
        self.add_feature(
            "degree assortativity",
            average_neighbor_degree,
            "average neighbor degree",
            InterpretabilityScore(4),
        )
