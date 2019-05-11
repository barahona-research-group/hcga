# -*- coding: utf-8 -*-
"""
Created on Sun Mar  3 18:30:46 2019

@author: Rob
"""

import pandas as pd
import numpy as np
from hcga.Operations.utils import clustering_quality

from networkx.algorithms.community import label_propagation_communities

class LabelpropagationCommunities():
    def __init__(self, G):
        self.G = G
        self.feature_names = []
        self.features = []

    def feature_extraction(self):

        """
        Identifies community sets determined by label propagation.
        """

        feature_names = ['node_ratio']

        G = self.G

        feature_list = []

        # basic normalisation parameters
        N = G.number_of_nodes()
        E = G.number_of_edges()

        c = list(label_propagation_communities(G))              

        
        # calculate ratio of the two communities
        feature_list.append((len(c[0])/len(c[1])))
        
        # clustering quality functions       
        qual_names,qual_vals = clustering_quality(G,c)           

            
        feature_list = feature_list + qual_vals
        feature_names = feature_names + qual_names     

        self.feature_names = feature_names
        self.features = feature_list
