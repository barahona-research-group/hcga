# -*- coding: utf-8 -*-
"""
Created on Sun Mar  3 18:30:46 2019

@author: Rob
"""

import pandas as pd
import numpy as np
from hcga.Operations.utils import clustering_quality
import networkx as nx

from networkx.algorithms.community import label_propagation_communities

class LabelpropagationCommunities():
    def __init__(self, G):
        self.G = G
        self.feature_names = []
        self.features = {}

    def feature_extraction(self):

        """
        Identifies community sets determined by label propagation.
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
    
            c = list(label_propagation_communities(G))              
    
            
            # calculate ratio of the two communities
            if len(c)>1:
                feature_list['node_ratio']=(len(c[0])/len(c[1]))
            else:
                feature_list['node_ratio']=0
            
            # clustering quality functions       
            qual_names,qual_vals = clustering_quality(G,c)  
            
            for i in range(len(qual_names)):
                feature_list[qual_names[i]]=qual_vals[i]         
    
            """ 
            feature_list = feature_list + qual_vals
            feature_names = feature_names + qual_names 
            """
        else:
            feature_list['node_ratio']=np.nan
            feature_list['node_ratio']=np.nan
            feature_list['mod']=np.nan
            feature_list['coverage']=np.nan
            feature_list['performance']=np.nan
            feature_list['inter_comm_edge']=np.nan
            feature_list['inter_comm_nedge']=np.nan
            feature_list['intra_comm_edge']=np.nan
            
            
        """
        self.feature_names = feature_names
        """
        self.features = feature_list
