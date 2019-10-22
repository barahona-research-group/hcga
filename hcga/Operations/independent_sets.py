# -*- coding: utf-8 -*-
"""
Created on Sun Mar  3 18:30:46 2019

@author: Rob
"""

import numpy as np
import networkx as nx
from hcga.Operations import utils


class IndependentSets():
    def __init__(self, G):
        self.G = G
        self.feature_names = []
        self.features = {}

    def feature_extraction(self):

        """Compute independent set measures.

        Parameters
        ----------
        G : graph
           A networkx graph

        Returns
        -------
        feature_list :list
           List of features related to independent sets.


        Notes
        -----


        """
        
        """
        self.feature_names = ['num_ind_nodes_norm','ratio__ind_nodes_norm']
        """

        G = self.G

        feature_list = {}
        
        if not nx.is_directed(G):            
            N = G.number_of_nodes() 
    
    
            ind_set = nx.maximal_independent_set(G)
            
            feature_list['num_ind_nodes_norm']=len(ind_set)
            
            feature_list['ratio__ind_nodes_norm']=len(ind_set)/len(G)
        else:
            feature_list['num_ind_nodes_norm']=np.nan
            feature_list['ratio__ind_nodes_norm']=np.nan

        self.features = feature_list
