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

featureclass_name = "Spectrum"


class Spectrum(FeatureClass):
    modes = ["medium", "slow"]
    shortname = "SPM"
    name = "spectrum"
    keywords = []
    normalize_features = True

    def compute_features(self):
        """Compute spectral measures.

        
        Notes
        -----

        
        References
        ----------


        """

        @lru_cache(maxsize=None)
        def eval_spectrum_adj(graph):
            return np.real(nx.linalg.spectrum.adjacency_spectrum(graph))

        # distribution of eigenvalues
        self.add_feature(
            "eigenvalues_adjacency",
            lambda graph: eval_spectrum_adj(graph),
            "The summary statistics of eigenvalues of adjacency matrix",
            InterpretabilityScore(3),
            statistics="centrality",
        )
        
        # get ratio of eigenvalues
        n_eigs = 10
        for i in range(n_eigs):
            for j in range(i):
                self.add_feature(
                    "eigenvalue_ratio_{}_{}".format(i,j),
                    lambda graph: eval_spectrum_adj(graph)[j]/eval_spectrum_adj(graph)[i],
                    "The ratio of the {} and {} eigenvalues".format(i,j),
                    InterpretabilityScore(2),
                )             
#        self.add_feature(
#            "eigenvalue_ratio_1_3",
#            lambda graph: eval_spectrum_adj(graph)[0]/eval_spectrum_adj(graph)[2],
#            "The ratio of the 1st and 3rd eigenvalues",
#            InterpretabilityScore(3),
#        )
#        self.add_feature(
#            "eigenvalue_ratio_1_4",
#            lambda graph: eval_spectrum_adj(graph)[0]/eval_spectrum_adj(graph)[3],
#            "The ratio of the 1st and 4th eigenvalues",
#            InterpretabilityScore(3),
#        )
#        self.add_feature(
#            "eigenvalue_ratio_1_5",
#            lambda graph: eval_spectrum_adj(graph)[0]/eval_spectrum_adj(graph)[4],
#            "The ratio of the 1st and 5th eigenvalues",
#            InterpretabilityScore(3),
#        )
#        self.add_feature(
#            "eigenvalue_ratio_2_3",
#            lambda graph: eval_spectrum_adj(graph)[1]/eval_spectrum_adj(graph)[2],
#            "The ratio of the 2st and 3nd eigenvalues",
#            InterpretabilityScore(3),
#        )       
        

        @lru_cache(maxsize=None)
        def eval_spectrum_modularity(graph):
            return np.real(nx.linalg.spectrum.modularity_spectrum(graph))

        # distribution of eigenvalues
        self.add_feature(
            "eigenvalues_modularity",
            lambda graph: eval_spectrum_modularity(graph),
            "The summary statistics of eigenvalues of modularity matrix",
            InterpretabilityScore(3),
            statistics="centrality",
        )

        @lru_cache(maxsize=None)
        def eval_spectrum_laplacian(graph):
            return np.real(nx.linalg.spectrum.laplacian_spectrum(graph))

        # distribution of eigenvalues
        self.add_feature(
            "eigenvalues_laplacian",
            lambda graph: eval_spectrum_laplacian(graph),
            "The summary statistics of eigenvalues of laplacian matrix",
            InterpretabilityScore(3),
            statistics="centrality",
        )
