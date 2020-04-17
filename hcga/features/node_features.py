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

from functools import lru_cache


from ..feature_class import FeatureClass, InterpretabilityScore

featureclass_name = "NodeFeatures"


class NodeFeatures(FeatureClass):
    modes = ["fast", "medium", "slow"]
    shortname = "NF"
    name = "node_features"
    keywords = []
    normalize_features = True

    def compute_features(self):
        """Compute node feature measures.

        


        Notes
        -----

        
        References
        ----------


        """

        
        @lru_cache(maxsize=None)
        def eval_node_feats(graph):
            # extracting feature matrix
            feature_matrix = []
            for node in graph.nodes:                
                feature_matrix.append(graph.nodes[node]['feat'])            
            return np.vstack(feature_matrix)

        @lru_cache(maxsize=None)
        def eval_conv_node_feats(graph):
            # extracting the convolved feature matrix - single message passing
            feature_matrix = []
            for node in graph.nodes:                
                feature_matrix.append(graph.nodes[node]['feat'])            
            return nx.to_numpy_array(graph).dot(np.vstack(feature_matrix))

        @lru_cache(maxsize=None)
        def eval_conv2_node_feats(graph):
            # extracting the convolved feature matrix - single message passing
            feature_matrix = []
            for node in graph.nodes:                
                feature_matrix.append(graph.nodes[node]['feat'])            
            return np.linalg.matrix_power(nx.to_numpy_array(graph),2).dot(np.vstack(feature_matrix))


                
        # compute statistics over each feature
        node_feat = lambda graph: eval_node_feats(graph)          
        self.add_feature(
            "node_feature_",
            node_feat,
            "The summary statistics of node feature ",
            InterpretabilityScore(5),
            statistics="node_features",
        )
 
        conv_node_feat = lambda graph: eval_conv_node_feats(graph)           
        self.add_feature(
            "conv_node_feature_",
            conv_node_feat,
            "The summary statistics after a single message passing of features of node feature ",
            InterpretabilityScore(3),
            statistics="node_features",
        )       

        conv2_node_feat = lambda graph: eval_conv2_node_feats(graph)           
        self.add_feature(
            "conv2_node_feature_",
            conv2_node_feat,
            "The summary statistics after a two message passing of features of node feature ",
            InterpretabilityScore(3),
            statistics="node_features",
        )    

        #TODO add more features based on node features here












