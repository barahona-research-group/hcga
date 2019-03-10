# -*- coding: utf-8 -*-
"""
Created on Sun Mar  3 18:30:46 2019

@author: Rob
"""

import pandas as pd
import numpy as np
import networkx as nx


def triangle_stats(G):
    """ 
    Input: Graph networkx
    
    Output: Dictionary of clustering stats
    
    Stats:
        number_of_nodes
        
    
    """
    
    triangles_stats_dict = {'number_of_triangles_mean':None,
            'number_of_triangles_mean_N-normalised':None,
            'number_of_triangles_mean_E-normalised':None,
            'graph_transivity':None,
            'graph_transivity_N-normalised':None,
            'graph_transivity_E-normalised':None,
            'graph_clustering_mean':None,
            'graph_clustering_N-normalised':None,
            'graph_clustering_E-normalised':None,
            'graph_clustering_std':None,
            'graph_clustering_median':None,
            'square_clustering_mean':None,
            'square_clustering_mean_N-normalised':None,
            'square_clustering_mean_E-normalised':None,
            'square_clustering_std':None,
            'square_clustering_median':None,
            }
    
    # basic normalisation parameters
    N = G.number_of_nodes() 
    E = G.number_of_edges()
    
    # Calculating number of triangles
    triangles_stats_dict['number_of_triangles_mean'] = np.asarray(list(nx.triangles(G).values())).mean()
    triangles_stats_dict['number_of_triangles_mean_N-normalised'] = np.asarray(list(nx.triangles(G).values())).mean()/N
    triangles_stats_dict['number_of_triangles_mean_E-normalised'] = np.asarray(list(nx.triangles(G).values())).mean()/E

    # graph transivity
    triangles_stats_dict['graph_transivity'] = nx.transitivity(G)
    triangles_stats_dict['graph_transivity_N-normalised'] = nx.transitivity(G)/N
    triangles_stats_dict['graph_transivity_E-normalised'] = nx.transitivity(G)/E
    
    # Average clustering coefficient
    triangles_stats_dict['graph_clustering_mean'] =  nx.average_clustering(G)
    triangles_stats_dict['graph_clustering_N-normalised'] =  nx.average_clustering(G)/N
    triangles_stats_dict['graph_clustering_E-normalised'] =  nx.average_clustering(G)/E
    triangles_stats_dict['graph_clustering_std'] == np.asarray(list(nx.clustering(G).values())).std()
    triangles_stats_dict['graph_clustering_median'] == np.median(np.asarray(list(nx.clustering(G).values())))
    
    # generalised degree
    triangles_stats_dict['square_clustering_mean'] = np.asarray(list(nx.square_clustering(G).values())).mean()
    triangles_stats_dict['square_clustering_mean_N-normalised'] = np.asarray(list(nx.square_clustering(G).values())).mean()/N
    triangles_stats_dict['square_clustering_mean_E-normalised'] = np.asarray(list(nx.square_clustering(G).values())).mean()/E
    triangles_stats_dict['square_clustering_std'] = np.asarray(list(nx.square_clustering(G).values())).std()
    triangles_stats_dict['square_clustering_median'] = np.median(np.asarray(list(nx.square_clustering(G).values())))   
    
    return triangles_stats_dict
   
   
    

