# -*- coding: utf-8 -*-
"""
Created on Sun Mar  3 18:30:46 2019

@author: Rob
"""

import numpy as np
import networkx as nx
from hcga.Operations import utils


class StructuralHoles():
    def __init__(self, G):
        self.G = G
        self.feature_names = []
        self.features = []

    def feature_extraction(self):

        r"""Compute structural hole measures.

        The *constraint* is a measure of the extent to which a node *v* is
        invested in those nodes that are themselves invested in the
        neighbors of *v*. Formally, the *constraint on v*, denoted `c(v)`,
        is defined by
    
        .. math::
    
           c(v) = \sum_{w \in N(v) \setminus \{v\}} \ell(v, w)
    
        where `N(v)` is the subset of the neighbors of `v` that are either
        predecessors or successors of `v` and `\ell(v, w)` is the local
        constraint on `v` with respect to `w` [1]_. For the definition of local
        constraint, see :func:`local_constraint`.
        
        Parameters
        ----------
        G : graph
           A networkx graph

        Returns
        -------
        feature_list :list
           List of features related to structural holes.


        Notes
        -----

        References
        ----------
        .. [1] Burt, Ronald S.
               "Structural holes and good ideas".
               American Journal of Sociology (110): 349–399.


        """

        self.feature_names = ['constraint_mean','constraint_std','constraint_median','constraint_max','constraint_min','constraint_opt_model',
                              'effective_size_mean','effective_size_std','effective_size_median','effective_size_max','effective_size_min','effective_size_opt_model']

        G = self.G

        feature_list = []

        N = G.number_of_nodes()
        
        constraint = list(nx.structuralholes.constraint(G).values())      
        
        # basic stats of constraint
        feature_list.append(np.mean(constraint))
        feature_list.append(np.mean(constraint))
        feature_list.append(np.median(constraint))
        feature_list.append(np.max(constraint))
        feature_list.append(np.min(constraint))
        
        # best distribution to fit data
        opt_mod_c,_ = utils.best_fit_distribution(constraint,bins=10)
        feature_list.append(opt_mod_c)

        effective_size = list(nx.structuralholes.effective_size(G).values())  
        
        # basic stats of effective size
        feature_list.append(np.mean(effective_size))
        feature_list.append(np.std(effective_size))
        feature_list.append(np.median(effective_size))
        feature_list.append(np.max(effective_size))
        feature_list.append(np.min(effective_size))


        # best distribution to fit data
        opt_mod_es,_ = utils.best_fit_distribution(constraint,bins=10)
        feature_list.append(opt_mod_es)
        
        self.features = feature_list
