# -*- coding: utf-8 -*-
"""
Created on Sun Mar  3 18:30:46 2019

@author: Rob
"""

import numpy as np
import networkx as nx

class ScaleFree():
    def __init__(self, G):
        self.G = G
        self.feature_names = []
        self.features = []

    def feature_extraction(self):

        """ Calculating metrics about scale-free networks

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

        self.feature_names = ['s_metric']

        G = self.G

        feature_list = []

       
        

        # 
        feature_list.append(nx.s_metric(G))


        
        

        self.features = feature_list
