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



    """
    feature_names = ['num_triangles','transitivity','clustering_mean',
                    'clustering_std','clustering_median','square_clustering_mean'
                    'square_clustering_std','square_clustering_median']
    feature_list = []

    # Calculating number of triangles
    feature_list.append(np.asarray(list(nx.triangles(G).values())).mean())

    # graph transivity
    feature_list.append(nx.transitivity(G))

    # Average clustering coefficient
    feature_list.append(nx.average_clustering(G))
    feature_list.append(np.asarray(list(nx.clustering(G).values())).std())
    feature_list.append(np.median(np.asarray(list(nx.clustering(G).values()))))

    # generalised degree
    feature_list.append(np.asarray(list(nx.square_clustering(G).values())).mean())
    feature_list.append(np.asarray(list(nx.square_clustering(G).values())).std())
    feature_list.append(np.median(np.asarray(list(nx.square_clustering(G).values()))))

    return (feature_names,feature_list)
