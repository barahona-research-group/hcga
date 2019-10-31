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

class kComponents():
    """
    k components class
    """
    def __init__(self, G):
        self.G = G
        self.feature_names = []
        self.features = {}

    def feature_extraction(self):

        """
        Compute features related to the k components of the network
        
        A k_component is a subgraph such that every node in it is connected to
        at least k other nodes in the subgraph.

        Parameters
        ----------
        G : graph
          A networkx graph

        Returns
        -------
        feature_list : list
           List of features related to k components.
        
        Notes
        -----
        K components calculations using networkx:
            `Networkx_kcomponents <https://networkx.github.io/documentation/stable/reference/algorithms/generated/networkx.algorithms.connectivity.kcomponents.k_components.html#networkx.algorithms.connectivity.kcomponents.k_components>`_            
        """
        

        G = self.G
        
        
        # This function is very slow for some graphs... 
        
        feature_list = {}
        if not nx.is_directed(G):            
            # Calculate the k_components
            k_components=nx.k_components(G)
            k_components_keys=np.asarray(list(k_components.keys()))
            k_components_vals=np.asarray(list(k_components.values()))
            
            # Calculate basic features related to k_components
            num_k=0
            for i in range(len(k_components_vals)):
                num_k=num_k+len(k_components_vals[i])
            
            max_k=max(k_components_keys)
            feature_list['num_k_components']=num_k
            feature_list['max_k']=max_k
            
            # Calculate basic feature related to largest k component
            feature_list['num_max_k_components']=len(k_components_vals[0])
            l=[len(i) for i in k_components_vals[0]]
            feature_list['mean_max_k_component_size']=np.mean(l)
            feature_list['largest_max_k_component']=max(l)
            feature_list['smallest_max_k_component']=min(l)
        
        else:
            feature_list['num_k_components']=np.nan
            feature_list['max_k']=np.nan
            feature_list['num_max_k_components']=np.nan
            feature_list['mean_max_k_component_size']=np.nan
            feature_list['largest_max_k_component']=np.nan
            feature_list['smallest_max_k_component']=np.nan
            

        self.features = feature_list
