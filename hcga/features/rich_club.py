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
import numpy as np

from functools import lru_cache

from ..feature_class import FeatureClass, InterpretabilityScore

featureclass_name = "RichClub"


class RichClub(FeatureClass):
    """ Rich Club class """

    modes = ["fast", "medium", "slow"]
    shortname = "RC"
    name = "rich_club"
    keywords = []
    normalize_features = True

    def compute_features(self):
        """
        Compute the rich club measures of the network

        Computed statistics
        -----
        Put here the list of things that are computed, with corresponding names

        """

        @lru_cache(maxsize=None)
        def eval_rich_club(graph):
            # extracting feature matrix       
            return list(nx.rich_club_coefficient(graph, normalized=False).values())

        # k = 1
        self.add_feature(
            "rich_club_k=1",
            lambda graph: eval_rich_club(graph)[0],
            "The rich-club coefficient is the ratio of the number of actual to the number of potential edges for nodes with degree greater than k",
            InterpretabilityScore(4),
        )

        # 
        self.add_feature(
            "rich_club_k=max",
            lambda graph: eval_rich_club(graph)[-1],
            "The rich-club coefficient is the ratio of the number of actual to the number of potential edges for nodes with degree greater than k",
            InterpretabilityScore(4),
        )
        
        self.add_feature(
            "rich_club_maxminratio",
            lambda graph: np.min(eval_rich_club(graph))/np.max(eval_rich_club(graph)),
            "The ratio of the smallest to largest rich club coefficients",
            InterpretabilityScore(4),
        )

        self.add_feature(
            "rich_club",
            lambda graph: eval_rich_club(graph),
            "The distribution of rich club coefficients",
            InterpretabilityScore(4),
            statistics="centrality",
        )
