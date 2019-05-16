# -*- coding: utf-8 -*-
"""
Created on Sun Mar  3 18:30:46 2019

@author: Rob
"""

import numpy as np
import networkx as nx
from hcga.Operations import utils


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

        self.feature_names = ['ratio_finite','closeness_mean','closeness_std',
                              'closeness_median','closeness_max',
                              'closeness_min']

        G = self.G

        feature_list = []

        N = G.number_of_nodes() 


        
        closeness_vitality_vals = np.asarray(list(nx.closeness_vitality(G).values()))      
        
        
        # remove infinites
        closeness_vitality_vals_fin  = closeness_vitality_vals[np.isfinite(closeness_vitality_vals)]
        
        # ratio of finite nodes to infinite vitality nodes
        ratio_finite = len(closeness_vitality_vals_fin)/len(closeness_vitality_vals)

        
        # standard measures of closeness vitality
        feature_list.append(np.mean(closeness_vitality_vals_fin))
        feature_list.append(np.std(closeness_vitality_vals_fin))
        feature_list.append(np.median(closeness_vitality_vals_fin))
        feature_list.append(np.max(closeness_vitality_vals_fin))
        feature_list.append(np.min(closeness_vitality_vals_fin))

        # fit distribution
        opt_mod,opt_mod_sse = utils.best_fit_distribution(closeness_vitality_vals_fin,bins=10)
        feature_list.append(opt_mod)        
        

        self.features = feature_list
