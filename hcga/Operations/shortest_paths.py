# -*- coding: utf-8 -*-
"""
Created on Sun Mar  3 18:30:46 2019

@author: Rob
"""

import pandas as pd
import numpy as np
import networkx as nx

class ShortestPaths():
    def __init__(self, G):
        self.G = G
        self.feature_names = []
        self.features = []

    def feature_extraction(self):

        """Compute the various shortest path measures.

        Parameters
        ----------
        G : graph
           A networkx graph

        Returns
        -------
        feature_list :list
           List of features related to shortest paths.


        Notes
        -----


        """
        
        
        """
        self.feature_names = ['path_length_mean','path_length_mean_max','path_length_max']
        """
        
        G = self.G

        feature_list = {}

        # Calculating the shortest paths stats
        shortest_paths = nx.shortest_path(G)

        # finding the mean and max shortest paths        
        shortest_path_length_mean = []
        shortest_path_length_max = []

        for i in G.nodes():
            shortest_path_length = []
            sp = list(shortest_paths[i].values())
            
            for j in G.nodes():
                shortest_path_length.append(len(sp[j]))
                
            shortest_path_length_mean.append(np.mean(shortest_path_length))
            shortest_path_length_max.append(np.max(shortest_path_length))

        feature_list['path_length_mean']=np.mean(shortest_path_length_mean)
        feature_list['path_length_mean_max']=np.mean(shortest_path_length_max)        
        feature_list['path_length_max']=np.max(shortest_path_length_max)       


        self.features = feature_list
