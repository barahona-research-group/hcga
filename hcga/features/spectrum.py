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

import numpy as np
import networkx as nx

from functools import lru_cache


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
 
        @lru_cache(maxsize=None)
        def eval_spectrum_modularity(graph):         
            return  np.real(nx.linalg.spectrum.modularity_spectrum(graph))
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
            return  np.real(nx.linalg.spectrum.laplacian_spectrum(graph))
        # distribution of eigenvalues
        self.add_feature(
            "eigenvalues_laplacian",
            lambda graph: eval_spectrum_laplacian(graph),
            "The summary statistics of eigenvalues of laplacian matrix",
            InterpretabilityScore(3),
            statistics="centrality",
        )
 



