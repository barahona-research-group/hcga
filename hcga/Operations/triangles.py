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

    triangles_stats = []

    # Calculating number of triangles
    triangles_stats.append(np.asarray(list(nx.triangles(G).values())).mean())


    # graph transivity
    triangles_stats.append(nx.transitivity(G))

    # Average clustering coefficient
    triangles_stats.append(nx.average_clustering(G))
    triangles_stats.append(np.asarray(list(nx.clustering(G).values())).std())
    triangles_stats.append(np.median(np.asarray(list(nx.clustering(G).values()))))

    # generalised degree
    triangles_stats.append(np.asarray(list(nx.square_clustering(G).values())).mean())
    triangles_stats.append(np.asarray(list(nx.square_clustering(G).values())).std())
    triangles_stats.append(np.median(np.asarray(list(nx.square_clustering(G).values()))))

    return triangles_stats
