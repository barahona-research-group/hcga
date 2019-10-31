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


import numpy as np
import networkx as nx
from hcga.Operations import utils


class StructuralHoles():
    def __init__(self, G):
        self.G = G
        self.feature_names = []
        self.features = {}

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
        

        
        G = self.G

        feature_list = {}

        
        constraint = list(nx.structuralholes.constraint(G).values())      
        
        try:
            # basic stats of constraint
            feature_list['constraint_mean']=np.mean(constraint)
            feature_list['constraint_std']=np.std(constraint)
            feature_list['constraint_median']=np.median(constraint)
            feature_list['constraint_max']=np.max(constraint)
            feature_list['constraint_min']=np.min(constraint)
        
            # best distribution to fit data
            opt_mod_c,_ = utils.best_fit_distribution(constraint,bins=10)
            feature_list['constraint_opt_model']=opt_mod_c
        
        except Exception as e:
            print('Exception for structural_holes', e)

            feature_list['constraint_mean']=np.nan
            feature_list['constraint_std']=np.nan
            feature_list['constraint_median']=np.nan
            feature_list['constraint_max']=np.nan
            feature_list['constraint_min']=np.nan
            feature_list['constraint_opt_model']=np.nan
            
        effective_size = list(nx.structuralholes.effective_size(G).values())  
        
        try:
            # basic stats of effective size
            feature_list['effective_size_mean']=np.mean(effective_size)
            feature_list['effective_size_std']=np.std(effective_size)
            feature_list['effective_size_median']=np.median(effective_size)
            feature_list['effective_size_max']=np.max(effective_size)
            feature_list['effective_size_min']=np.min(effective_size)
    
    
            # best distribution to fit data
            opt_mod_es,_ = utils.best_fit_distribution(constraint,bins=10)
            feature_list['effective_size_opt_model']=opt_mod_es
            
        except Exception as e:
            print('Exception for structural_holes (2nd)', e)

            feature_list['effective_size_mean']=np.nan
            feature_list['effective_size_std']=np.nan
            feature_list['effective_size_median']=np.nan
            feature_list['effective_size_max']=np.nan
            feature_list['effective_size_min']=np.nan
            feature_list['effective_size_opt_model']=np.nan
            
        self.features = feature_list
