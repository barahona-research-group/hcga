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

from functools import lru_cache

import networkx as nx
import numpy as np

from ..feature_class import FeatureClass, InterpretabilityScore

featureclass_name = "NodeFeatures"


class NodeFeatures(FeatureClass):
    modes = ["fast", "medium", "slow"]
    shortname = "NF"
    name = "node_features"
    encoding = "networkx"

    def compute_features(self):
        """Compute node feature measures.

        


        Notes
        -----

        
        References
        ----------


        """

        @lru_cache(maxsize=None)
        def get_feature_matrix(graph):
            """Extracting feature matrix."""
            return np.vstack([graph.nodes[node]["feat"] for node in graph.nodes])

        @lru_cache(maxsize=None)
        def get_conv_matrix(graph):
            """Extracting feature matrix."""
            return nx.to_numpy_array(graph) + np.eye(len(graph))

        self.add_feature(
            "node_feature",
            lambda graph: get_feature_matrix(graph),
            "The summary statistics of node feature ",
            InterpretabilityScore(5),
            statistics="node_features",
        )

        self.add_feature(
            "conv_node_feature",
            lambda graph: get_conv_matrix(graph).dot(get_feature_matrix(graph)),
            "The summary statistics after a single message passing of features of node feature ",
            InterpretabilityScore(3),
            statistics="node_features",
        )

        self.add_feature(
            "conv2_node_feature",
            lambda graph: np.linalg.matrix_power(get_conv_matrix(graph), 2).dot(
                get_feature_matrix(graph)
            ),
            "The summary statistics after a two message passing of features of node feature ",
            InterpretabilityScore(3),
            statistics="node_features",
        )

        # TODO add more features based on node features here
        self.add_feature(
            "mean_all_node_features",
            lambda graph: np.mean(get_feature_matrix(graph), axis=1).tolist(),
            "The summary statistics of the mean of all node features ",
            InterpretabilityScore(3),
            statistics="centrality",
        )

        self.add_feature(
            "mean_node_feature",
            lambda graph: np.mean(get_feature_matrix(graph), axis=0).tolist(),
            "The summary statistics of the mean of each feature across the nodes ",
            InterpretabilityScore(3),
            statistics="centrality",
        )

        self.add_feature(
            "max_node_feature",
            lambda graph: np.max(get_feature_matrix(graph), axis=1).tolist(),
            "The summary statistics of the max of all node features ",
            InterpretabilityScore(3),
            statistics="centrality",
        )

        self.add_feature(
            "max_node_feature",
            lambda graph: np.max(get_feature_matrix(graph), axis=0).tolist(),
            "The summary statistics of the max of each feature across the nodes ",
            InterpretabilityScore(3),
            statistics="centrality",
        )

        self.add_feature(
            "min_node_feature",
            lambda graph: np.min(get_feature_matrix(graph), axis=1).tolist(),
            "The summary statistics of the min of all node features ",
            InterpretabilityScore(3),
            statistics="centrality",
        )
        self.add_feature(
            "min_node_feature",
            lambda graph: np.min(get_feature_matrix(graph), axis=0).tolist(),
            "The summary statistics of the min of each feature across the nodes ",
            InterpretabilityScore(3),
            statistics="centrality",
        )

        self.add_feature(
            "sum_node_feature",
            lambda graph: np.sum(get_feature_matrix(graph), axis=1).tolist(),
            "The summary statistics of the sum of all node features ",
            InterpretabilityScore(3),
            statistics="centrality",
        )
        self.add_feature(
            "sum_nde_feature",
            lambda graph: np.sum(get_feature_matrix(graph), axis=0).tolist(),
            "The summary statistics of the sum of each feature across the nodes ",
            InterpretabilityScore(3),
            statistics="centrality",
        )
