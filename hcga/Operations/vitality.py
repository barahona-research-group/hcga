# -*- coding: utf-8 -*-
"""
Created on Sun Mar  3 18:30:46 2019

@author: Rob
"""

import numpy as np
import networkx as nx

class Vitality():
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

        self.feature_names = ['closeness_mean','closeness_std',
                              'closeness_median','closeness_max',
                              'closeness_min']

        G = self.G

        feature_list = []

        N = G.number_of_nodes()
        
        closeness_vitality_vals = np.asarray(list(nx.closeness_vitality(G).values()))
        
        # remove infinites
        closeness_vitality_vals  = closeness_vitality_vals[np.isfinite(closeness_vitality_vals)]
        
        # 
        feature_list.append(np.mean(closeness_vitality_vals))
        feature_list.append(np.std(closeness_vitality_vals))
        feature_list.append(np.median(closeness_vitality_vals))
        feature_list.append(np.max(closeness_vitality_vals))
        feature_list.append(np.min(closeness_vitality_vals))

        
        

        self.features = feature_list
