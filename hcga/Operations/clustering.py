# -*- coding: utf-8 -*-
"""
Created on Sun Mar  3 18:30:46 2019

@author: Rob
"""

import pandas as pd
import numpy as np


def clustering_stats(G):
    """ 
    Input: Graph networkx
    
    Output: Dictionary of clustering stats
    
    Stats:
        number_of_nodes
        
    
    """
    
    basic_stats_dict = {'number_of_nodes':None,
                        'number_of_edges':None,
                        'degree_mean':None,
                        'degree_median':None,
                        'degree_std':None                        
                        }

        

    basic_stats_dict['number_of_nodes'] = G.number_of_nodes()    
    basic_stats_dict['number_of_edges'] = G.number_of_edges()
    
    
    # Degree stats
    degree_vals = np.asarray(list(G.degree().values()))
    
    basic_stats_dict['degree_mean'] = degree_vals.mean()
    basic_stats_dict['degree_median'] = degree_vals.median()
    basic_stats_dict['degree_std'] = degree_vals.std()
    
