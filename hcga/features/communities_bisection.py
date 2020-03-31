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


from networkx.algorithms.community import kernighan_lin_bisection

from .feature_class import FeatureClass, InterpretabilityScore

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

        self.add_feature(
            "partition",
            lambda graph: list(kernighan_lin_bisection(graph)),
            "The optimal partition after async fluid optimisations for c={}",
            InterpretabilityScore(4),
        )
