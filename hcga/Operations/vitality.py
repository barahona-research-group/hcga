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


class Vitality():
    def __init__(self, G):
        self.G = G
        self.feature_names = []
        self.features = {}

    def feature_extraction(self):

        """Compute node connectivity measures.

        Parameters
        ----------
        G : graph
           A networkx graph

        Returns
        -------
        feature_list :list
           List of features related to node connectivity.


        Notes
        -----


        """
        

        

        G = self.G

        feature_list = {}



        
        closeness_vitality_vals = np.asarray(list(nx.closeness_vitality(G).values()))      
        
        
        try:
            # remove infinites
            closeness_vitality_vals_fin  = closeness_vitality_vals[np.isfinite(closeness_vitality_vals)]
            
            # ratio of finite nodes to infinite vitality nodes
            ratio_finite = len(closeness_vitality_vals_fin)/len(closeness_vitality_vals)
            
            try:
                # standard measures of closeness vitality
                feature_list['closeness_mean']=np.mean(closeness_vitality_vals_fin)
                feature_list['closeness_std']=np.std(closeness_vitality_vals_fin)
                feature_list['closeness_median']=np.median(closeness_vitality_vals_fin)
                feature_list['closeness_max']=np.max(closeness_vitality_vals_fin)
                feature_list['closeness_min']=np.min(closeness_vitality_vals_fin)
        
                # fit distribution
                opt_mod,opt_mod_sse = utils.best_fit_distribution(closeness_vitality_vals_fin,bins=10)
                feature_list['opt_mod']=opt_mod  

            except Exception as e:
                print('Exception for vitality', e)
                feature_list['closeness_mean']=np.nan
                feature_list['closeness_std']=np.nan
                feature_list['closeness_median']=np.nan
                feature_list['closeness_max']=np.nan
                feature_list['closeness_min']=np.nan
                feature_list['opt_mod']=np.nan
        
        except Exception as e:
            print('Exception for vitality (2nd)', e)
            feature_list['closeness_mean']=np.nan
            feature_list['closeness_std']=np.nan
            feature_list['closeness_median']=np.nan
            feature_list['closeness_max']=np.nan
            feature_list['closeness_min']=np.nan
            feature_list['opt_mod']=np.nan

        self.features = feature_list
