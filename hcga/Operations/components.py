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

class Components():
    """
    Components class
    """
 
    def __init__(self, G):
        self.G = G
        self.feature_names = []
        self.features = {}

    def feature_extraction(self):
        """Compute the component features for the network


        Parameters
        ----------
        G : graph
          A networkx graph

        
        Returns
        -------
        feature_list : dict
           Dictionary of features related to components.


        Notes
        -----
        Components calculations using networkx:
            `Networkx_components <https://networkx.github.io/documentation/stable/reference/algorithms/component.html>`_
        """
        

        G = self.G

        feature_list = {}
        
        if nx.is_directed(G):
            
            # Compute component features
            feature_list['num_strongly_conn_comps']=nx.number_strongly_connected_components(G)
            feature_list['num_weak_conn_comps']=nx.number_weakly_connected_components(G)
            feature_list['num_attracting_comps']=nx.number_attracting_components(G)
            
        else:
            feature_list['num_strongly_conn_comps']=np.nan
            feature_list['num_weak_conn_comps']=np.nan
            feature_list['num_attracting_comps']=np.nan
    
        
        
        self.features = feature_list
