# -*- coding: utf-8 -*-
"""
Created on Sun Mar  3 18:30:46 2019

@author: Rob
"""

import pandas as pd
import numpy as np
import networkx as nx

class RichClub():
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
        self.feature_names = ['num_rich','mean_rich_coef','std_rich_coef','max_rich_coef','ratio_rich_coef','ratio_top2_coef',
                              'top10_10','top10_9','top10_8','top10_7','top10_6','top10_5','top10_4','top10_3',
                              'top10_2','top10_1']
        """
        

        G = self.G

        feature_list = {}
        if not nx.is_directed(G):            
            # Calculating the shortest paths stats
            try:
                rich_club = list(nx.rich_club_coefficient(G).values())       
                
                # calculate number of nodes that qualify according to degree k
                feature_list['num_rich']=len(rich_club)
                
                feature_list['mean_rich_coef']=np.mean(rich_club)
                feature_list['std_rich_coef']=np.std(rich_club)
                feature_list['max_rich_coef']=np.max(rich_club)      
                
                feature_list['ratio_rich_coef']=np.min(rich_club)/np.max(rich_club)
                
                if rich_club[-2]>0:
                    feature_list['ratio_top2_coef']=rich_club[-1]/rich_club[-2]
                else:
                    feature_list['ratio_top2_coef']=0
        
                # top ten degree rich club coefficients
                if len(rich_club)>=10:
                    top10 = rich_club[-10:]
                else:
                    top10 = [1] * (10-len(rich_club)) + rich_club
                    
                """
                feature_list = feature_list + top10
                """
                
                l=[10,9,8,7,6,5,4,3,2,1]
                for i in range(len(top10)):
                    feature_list['top10_{}'.format(l[i])]=top10[i]
                
                    
            except:
                feature_names = ['num_rich','mean_rich_coef','std_rich_coef','max_rich_coef','ratio_rich_coef','ratio_top2_coef',
                                  'top10_10','top10_9','top10_8','top10_7','top10_6','top10_5','top10_4','top10_3',
                                  'top10_2','top10_1']
                for j in range(len(feature_names)):
                    feature_list[feature_names[j]]=0
        else:
            feature_list['rich_club_features']='unavailable for directed graphs'
            
        """
        feature_list = [0] * 16
        """
            
        self.features = feature_list
