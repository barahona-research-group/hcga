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


import networkx as nx

class ScaleFree():
    """
    Scale free class
    
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
           Dictionary of features related to scale freeness.


        Notes
        -----
                Scale free calculations using networkx:
            `Networkx_scale free <https://networkx.github.io/documentation/stable/_modules/networkx/algorithms/smetric.html#s_metric>`_
        
        The s-metric is defined as the sum of the products deg(u)*deg(v)
        for every edge (u,v) in G. If norm is provided construct the
        s-max graph and compute it's s_metric, and return the normalized
        s value      
        
        
        References
        ----------
        .. [1] Lun Li, David Alderson, John C. Doyle, and Walter Willinger,
               Towards a Theory of Scale-Free Graphs:
               Definition, Properties, and  Implications (Extended Version), 2005.
               https://arxiv.org/abs/cond-mat/0501169

        """

        G = self.G

        feature_list = {}

              

        # 
        feature_list['s_metric']=nx.s_metric(G,normalized=False)        
        

        self.features = feature_list
