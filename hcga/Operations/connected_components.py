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

class ConnectedComponents():
    """
    Connected Components class
    """
    def __init__(self, G):
        self.G = G
        self.feature_names = []
        self.features = {}

    def feature_extraction(self):

        """ Calculating metrics about scale-free networks

        Parameters
        ----------
        G : graph
           A networkx graph

        Returns
        -------
        feature_list : dict
           Dictionary of features related to node connectivity.


        Notes
        -----
        Components calculations using networkx:
            `Networkx_connected_components <https://networkx.github.io/documentation/stable/reference/algorithms/generated/networkx.algorithms.components.connected_components.html#networkx.algorithms.components.connected_components>`_


        """
        


        G = self.G

        feature_list = {}

              
        if not nx.is_connected(G): 
            conn_components = list(nx.connected_components(G)) 
            feature_list['is_connected']=0
            feature_list['num_conncomp']=len(conn_components)
            feature_list['ratio_conncomp_size']=len(conn_components[0])/len(conn_components[1])
            feature_list['ratio_conncomp_size_max_min']=len(conn_components[0])/len(conn_components[-1])
        else:
            feature_list['is_connected']=1
            feature_list['num_conncomp']=np.nan   
            feature_list['ratio_conncomp_size']=np.nan
            feature_list['ratio_conncomp_size_max_min']=np.nan

        self.features = feature_list
