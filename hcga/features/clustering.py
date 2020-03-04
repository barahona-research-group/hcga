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
import networkx as nx

from .feature_class import FeatureClass
from .feature_class import InterpretabilityScore
from ..feature_utils import summary_statistics


featureclass_name = 'Clustering'

class Clustering(FeatureClass):
    """
    Clustering class
    """    

    modes = ['medium', 'slow']
    shortname = 'CL'
    name = 'clustering'
    keywords = []
    normalize_features = True
    

    def compute_features(self):
        """Compute the various clustering measures.

        Notes
        -----
        Implementation of networkx code:
            `Networkx_clustering <https://networkx.github.io/documentation/stable/reference/algorithms/clustering.html>`_

        We followed the same structure as networkx for implementing clustering features.

        """
        
        if not nx.is_directed(self.graph):
            triang = np.asarray(list(nx.triangles(self.graph).values())).mean()
            transi = nx.transitivity(self.graph)
        else:
            transi = np.nan
            triang = np.nan

        self.add_feature('num_triangles', triang, 
            'Number of triangles in the graph', 
            InterpretabilityScore('max'))
        self.add_feature('transitivity', transi, 
            'Transitivity of the graph', 
            InterpretabilityScore('max'))

        # Average clustering coefficient
        clustering_dist = list(nx.clustering(self.graph).values())
        summary_statistics(self.add_feature, clustering_dist, 
                'clustering', 'the clustering of the graph', 
                InterpretabilityScore('max'))       

        # generalised degree
        square_clustering_dist = list(nx.square_clustering(self.graph).values())
        summary_statistics(self.add_feature, square_clustering_dist, 
                'square_clustering', 'the square clustering of the graph', 
                InterpretabilityScore('max'))       
