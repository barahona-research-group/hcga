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

featureclass_name = "IndependentSets"


class IndependentSets(FeatureClass):
    """Independent sets class
    


    Parameters
    ----------



    Notes
    -----


    References
    ----------

    
    """

    modes = ["fast", "medium", "slow"]
    shortname = "IS"
    name = "independent_sets"
    encoding = "networkx"

    def compute_features(self):
        """
        Compute the independent sets of the network

        Computed statistics
        -----
        Put here the list of things that are computed, with corresponding names

        """

        self.add_feature(
            "size_max_indep_set",
            lambda graph: len(nx.maximal_independent_set(graph)),
            "The number of nodes in the maximal independent set",
            InterpretabilityScore(3),
        )
