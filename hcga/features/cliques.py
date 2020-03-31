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

from networkx.algorithms import clique

from .feature_class import FeatureClass, InterpretabilityScore

featureclass_name = "Cliques"


class Cliques(FeatureClass):
    """Basic stats class"""

    modes = ["fast", "medium", "slow"]
    shortname = "Cli"
    name = "cliques"
    keywords = []
    normalize_features = True

    def compute_features(self):
        """
        Compute some clique based measures for the network

        Computed statistics
        -----
        Put here the list of things that are computed, with corresponding names

        """

        # graph clique number
        self.add_feature(
            "graph_clique_number",
            lambda graph: clique.graph_clique_number(graph),
            "The clique number of a graph is the size of the largest clique in the graph",
            InterpretabilityScore(3),
        )

        # number of maximal cliques
        self.add_feature(
            "num_max_cliques",
            lambda graph: clique.graph_number_of_cliques(graph),
            "The number of maximal cliques in the graph",
            InterpretabilityScore(3),
        )

        n_cliques = lambda graph: len(
            [u for u in list(clique.enumerate_all_cliques(graph)) if len(u) > 1]
        )
        self.add_feature(
            "num_cliques",
            n_cliques,
            "The number of cliques in the graph",
            InterpretabilityScore(3),
        )

        def clique_sizes(graph):
            out = [len(u) for u in list(clique.enumerate_all_cliques(graph)) if len(u) > 1]
            if len(out) == 0:
                return [np.nan]
            return out
        self.add_feature(
            "clique sizes",
            clique_sizes,
            "the distribution of clique sizes",
            InterpretabilityScore(3),
        )
