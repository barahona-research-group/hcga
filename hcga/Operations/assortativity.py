# -*- coding: utf-8 -*-
"""
Created on Sun Mar  3 18:30:46 2019

@author: Rob
"""

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
