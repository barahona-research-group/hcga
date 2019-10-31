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

class MinimumCuts():
    """
    Minimum cuts class
    """
    def __init__(self, G):
        self.G = G
        self.feature_names = []
        self.features = {}

    def feature_extraction(self):

        """Compute the minimum cuts for the network
        

        Parameters
        ----------
        G : graph
          A networkx graph

        Returns
        -------
        feature_list : list
           List of features related to theminimum cuts.
           
        Notes
        -----
        Calculations using networkx:
            `Networkx_minimum_cuts <https://networkx.github.io/documentation/stable/reference/algorithms/connectivity.html>`_
        
        """
        
        G = self.G

        feature_list = {}
        
        #Compute minimum cuts
        feature_list['min_node_cut_size']=len(nx.minimum_node_cut(G))
        feature_list['min_edge_cut_size']=len(nx.minimum_edge_cut(G))
        
        self.features = feature_list
