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

class Assortativity():
    """
    Assortativity class
    """
    def __init__(self, G):
        self.G = G
        self.feature_names = []
        self.features = {}

    def feature_extraction(self):

        """
        Compute assortativity measures.
        
        Assortativity measures the number of connections that nodes make with 
        other nodes that are similar.


        Parameters
        ----------
        G : graph
           A networkx graph

        Returns
        -------
        feature_list :list
           List of features related to assortativity.


        Notes
        -----
        Implementation of networkx code:
            `Networkx_assortativity <https://networkx.github.io/documentation/latest/reference/algorithms/assortativity.html>`_
        
        """

        G = self.G

        feature_list = {}

        N = G.number_of_nodes()
        
        # Compute some assortativity features
        feature_list['degree_assortativity_coeff'] = nx.degree_assortativity_coefficient(G)
        feature_list['degree_pearson_corr_coef'] = nx.degree_pearson_correlation_coefficient(G)
        

        self.features = feature_list
