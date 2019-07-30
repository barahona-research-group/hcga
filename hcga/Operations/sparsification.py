# -*- coding: utf-8 -*-
"""
Created on Sun Mar  3 18:30:46 2019

@author: Rob
"""

import numpy as np
import networkx as nx
from hcga.Operations import utils


class Sparsification():
    def __init__(self, G):
        self.G = G
        self.feature_names = []
        self.features = {}

    def feature_extraction(self):

        r"""
        
        Parameters
        ----------
        G : graph
           A networkx graph

        Returns
        -------
        feature_list :list
           List of features related to sparsification.


        Notes
        -----

        References
        ----------
        .. [1] Burt, Ronald S.
               "Structural holes and good ideas".
               American Journal of Sociology (110): 349–399.


        """

        """
        self.feature_names = ['']
        """
        
        G = self.G

        feature_list = {}

        N = G.number_of_nodes()       

       
        self.features = feature_list
