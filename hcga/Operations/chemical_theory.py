# -*- coding: utf-8 -*-
"""
Created on Sun Mar  3 18:30:46 2019

@author: Rob
"""

import numpy as np
import networkx as nx

class ChemicalTheory():
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


        """

        self.feature_names = ['wiener_index']

        G = self.G

        feature_list = []

        N = G.number_of_nodes()
        

        # calculate wiener index using networkx
        feature_list.append(nx.wiener_index(G))
        
        
        

        self.features = feature_list
