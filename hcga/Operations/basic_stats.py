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
    feature_names = ['num_nodes','num_edges','degree_mean','degree_median','degree_std']
    feature_list = []


    # basic normalisation parameters
    N = G.number_of_nodes()
    E = G.number_of_edges()

    # Adding basic node and edge numbers
    feature_list.append(N)
    feature_list.append(E)

    # Degree stats
    degree_vals = np.asarray(list(dict(G.degree())))

    feature_list.append(degree_vals.mean())
    feature_list.append(np.median(degree_vals))
    feature_list.append(degree_vals.std())

    return (feature_names,feature_list)
