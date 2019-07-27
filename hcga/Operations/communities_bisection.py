# -*- coding: utf-8 -*-
"""
Created on Sun Mar  3 18:30:46 2019

@author: Rob
"""

import pandas as pd
import numpy as np
from hcga.Operations.utils import clustering_quality
import networkx as nx

from networkx.algorithms.community import kernighan_lin_bisection

class BisectionCommunities():
    def __init__(self, G):
        self.G = G
        self.feature_names = []
        self.features = []

    def feature_extraction(self):

        """
        
        """
        
        """
        feature_names = ['node_ratio']
        """

        G = self.G

        feature_list = {}
        
        if not nx.is_directed(G):
            # basic normalisation parameters
            N = G.number_of_nodes()
            E = G.number_of_edges()

            c = list(kernighan_lin_bisection(G))        
        

        
            # calculate ratio of the two communities
            feature_list['node_ratio']=(len(c[0])/len(c[1]))
        
            # clustering quality functions       
            qual_names,qual_vals = clustering_quality(G,c)           

            for i in range(len(qual_names)):
                feature_list[qual_names[i]]=qual_vals[i]
            
            """
            feature_list = feature_list + qual_vals
            feature_names = feature_names + qual_names     
            """
        else:
            feature_list['bisection_features']='unavailable for directed graphs'

        """
        self.feature_names = feature_names
        """
        self.features = feature_list
