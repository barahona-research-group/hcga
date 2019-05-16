# -*- coding: utf-8 -*-
"""
Created on Sun Mar  3 18:30:46 2019

@author: Rob
"""

import numpy as np
import networkx as nx

class Assortativity():
    def __init__(self, G):
        self.G = G
        self.feature_names = []
        self.features = []

    def feature_extraction(self):

        """Compute node connectivity measures.

        Parameters
        ----------
        G : graph
           A networkx graph

        Returns
        -------
        feature_list :list
           List of features related to node connectivity.


        Notes
        -----
        Implementation of networkx code:
            https://networkx.github.io/documentation/latest/reference/algorithms/assortativity.html
        

        """

        self.feature_names = ['degree_assortativity_coeff',
                              'degree_pearson_corr_coef']

        G = self.G

        feature_list = []

        N = G.number_of_nodes()
        

        feature_list.append(nx.degree_assortativity_coefficient(G))
        feature_list.append(nx.degree_pearson_correlation_coefficient(G))
        
        
        

        self.features = feature_list
