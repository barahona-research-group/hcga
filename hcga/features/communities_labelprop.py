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

from networkx.algorithms.community import label_propagation_communities

from ..feature_class import FeatureClass, InterpretabilityScore

featureclass_name = "CommunitiesLabelPropagation"


class CommunitiesLabelPropagation(FeatureClass):
    """
    Communities Label propagation class
    """

    modes = ["medium", "slow"]
    shortname = "CLP"
    name = "communities_labelprop"
    keywords = []
    normalize_features = True

    def compute_features(self):
        """Compute the measures about community detection using label propagation algorithm.

        Notes
        -----
        """

        @lru_cache(maxsize=None)
        def eval_labelprop(graph):
            """this evaluates the main function and cach it for speed up"""
            communities = list(label_propagation_communities(graph))

            # if a single communities, add a trivial one
            if len(communities) == 1:
                communities.append([{0}])
            return communities

        self.add_feature(
            "ratio_commsize",
            lambda graph: len(eval_labelprop(graph)[0]) / len(eval_labelprop(graph)[1]),
            "The ratio of the largest and second largest communities using label propagation",
            InterpretabilityScore(4),
        )

        self.add_feature(
            "sum_density_c={}",
            lambda graph: eval_labelprop(graph),
            "The optimal partition using label propagation algorithm",
            InterpretabilityScore(4),
            statistics="clustering",
        )
