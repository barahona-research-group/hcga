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
import numpy as np

from ..feature_class import FeatureClass, InterpretabilityScore

featureclass_name = "CycleBasis"


class CycleBasis(FeatureClass):
    """Cycle Basis class"""

    modes = ["fast", "medium", "slow"]
    shortname = "CYB"
    name = "cycle_basis"
    keywords = []
    normalize_features = True

    def compute_features(self):
        """
        Compute the cycle basis of the network

        Computed statistics
        -----
        Put here the list of things that are computed, with corresponding names

        """

        @lru_cache(maxsize=None)
        def eval_cycle_basis(graph):
            """this evaluates the main function and cach it for speed up"""
            return nx.cycle_basis(graph)

        self.add_feature(
            "num_cycles",
            lambda graph: len(eval_cycle_basis(graph)),
            "The total number of cycles in the graph",
            InterpretabilityScore(3),
        )

        self.add_feature(
            "average_cycle_length",
            lambda graph: np.mean([len(l) for l in eval_cycle_basis(graph)]),
            "The average length of cycles in the graph",
            InterpretabilityScore(3),
        )

        self.add_feature(
            "minimum_cycle_length",
            lambda graph: np.min([len(l) for l in eval_cycle_basis(graph)]),
            "The minimum length of cycles in the graph",
            InterpretabilityScore(3),
        )

        ratio_nodes_cycle = lambda graph: len(
            np.unique([x for l in eval_cycle_basis(graph) for x in l])
        ) / len(graph)
        self.add_feature(
            "ratio_nodes_cycle",
            ratio_nodes_cycle,
            "The ratio of nodes that appear in at least one cycle to the total number of nodes in the graph",
            InterpretabilityScore(3),
        )
