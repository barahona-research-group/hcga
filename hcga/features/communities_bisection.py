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

from networkx.algorithms.community import kernighan_lin_bisection

from ..feature_class import FeatureClass, InterpretabilityScore

featureclass_name = "CommunitiesBisection"


class CommunitiesBisection(FeatureClass):
    """
    Communities Bisection class
    """

    modes = ["medium", "slow"]
    shortname = "CBI"
    name = "communities_bisection"
    keywords = []
    normalize_features = True

    def compute_features(self):
        """Compute the measures about community detection using bisection algorithm.

        Notes
        -----
        """

        @lru_cache(maxsize=None)
        def eval_bisection(graph):
            """this evaluates the main function and cach it for speed up"""
            communities = list(kernighan_lin_bisection(graph))

            # if a single communities, add a trivial one
            if len(communities) == 1:
                communities.append([{0}])

            # sort sets by size
            communities.sort(key=len, reverse=True)

            return communities

        self.add_feature(
            "largest_commsize",
            lambda graph: len(eval_bisection(graph)[0]),
            "The ratio of the largest and second largest communities using bisection algorithm",
            InterpretabilityScore(4),
        )

        self.add_feature(
            "ratio_commsize",
            lambda graph: len(eval_bisection(graph)[0]) / len(eval_bisection(graph)[1]),
            "The ratio of the largest and second largest communities using bisection algorithm",
            InterpretabilityScore(3),
        )

        self.add_feature(
            "partition",
            eval_bisection,
            "The optimal partition for kernighan lin bisection algorithm",
            InterpretabilityScore(4),
            statistics="clustering",
        )
