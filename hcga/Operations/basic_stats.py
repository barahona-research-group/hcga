# -*- coding: utf-8 -*-
"""
Created on Sun Mar  3 18:30:46 2019

@author: Rob
"""

import pandas as pd
import numpy as np


def basic_stats(G):
    """ 
    Input: Graph networkx
    
    Output: Dictionary of Basic graph stats
    
    Stats:
        number_of_nodes
        
    
    """
    
    basic_stats_dict = {'number_of_nodes':None,
                        'number_of_edges':None,
                        'degree_mean':None,
                        'degree_median':None,
                        'degree_std':None                        
                        }

        
    # basic normalisation parameters
    N = G.number_of_nodes() 
    E = G.number_of_edges()

    basic_stats_dict['number_of_nodes'] = N  
    basic_stats_dict['number_of_edges'] = E
    
    
    # Degree stats
    degree_vals = np.asarray(list(dict(G.degree())))
    
    basic_stats_dict['degree_mean'] = degree_vals.mean()
    basic_stats_dict['degree_mean_N-normalised'] = degree_vals.mean()/N
    basic_stats_dict['degree_mean_E-normalised'] = degree_vals.mean()/E
    basic_stats_dict['degree_median'] = np.median(degree_vals)
    basic_stats_dict['degree_std'] = degree_vals.std()
    
    return basic_stats_dict
    
