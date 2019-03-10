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

    basic_stats = []


    # basic normalisation parameters
    N = G.number_of_nodes()
    E = G.number_of_edges()

    # Adding basic node and edge numbers
    basic_stats.append(N)
    basic_stats.append(E)

    # Degree stats
    degree_vals = np.asarray(list(dict(G.degree())))

    basic_stats.append(degree_vals.mean())
    basic_stats.append(np.median(degree_vals))
    basic_stats.append(degree_vals.std())

    return basic_stats
