#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 28 16:51:19 2019

@author: Henry
"""

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
        feature_list :list
           List of features related to components.


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