#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 27 19:07:39 2019

@author: Henry
"""

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