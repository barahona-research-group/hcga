# -*- coding: utf-8 -*-
# This file is part of hcga.
#
# Copyright (C) 2019, 
# Robert Peach (r.peach13@imperial.ac.uk), 
# Alexis Arnaudon (alexis.arnaudon@epfl.ch), 
# https://github.com/ImperialCollegeLondon/hcga.git
#
# hcga is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# hcga is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with hcga.  If not, see <http://www.gnu.org/licenses/>.


import pandas as pd
import numpy as np
from hcga.Operations.utils import summary_statistics

class BasicStats():
    """
    Basic stats class
    """
    def __init__(self, G):
        self.G = G
        self.feature_names = []
        self.features = {}

    def feature_extraction(self):

        """
        Compute some basic stats of the network


        Parameters
        ----------
        G : graph
          A networkx graph

        Returns
        -------
        feature_list : list
           List of features related to basic stats.
            
        Notes
        -----
        Basic stats calculations using networkx:
            `Networkx_basic_stats <https://networkx.github.io/documentation/stable/reference/functions.html>`_
        """


        G = self.G

        feature_list = {}

        # basic normalisation parameters
        N = G.number_of_nodes()
        E = G.number_of_edges()

        # Adding basic node and edge numbers
        feature_list['num_nodes'] = N
        feature_list['num_edges'] = E

        # Degree stats
        degree_vals = list(dict(G.degree()).values())
        
        feature_list = summary_statistics(feature_list,degree_vals,'degree')       

        #feature_list['degree_mean'] = degree_vals.mean()
        #feature_list['degree_median'] = np.median(degree_vals)
        #feature_list['degree_std'] = degree_vals.std()
        
        
        feature_list['density'] = 2*E / (N*(N-1))

        self.features = feature_list



"""
def basic_stats(G):

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
"""
