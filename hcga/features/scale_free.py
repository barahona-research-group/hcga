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

from ..feature_class import FeatureClass, InterpretabilityScore

featureclass_name = "ScaleFree"


class ScaleFree(FeatureClass):
    """ Scale Free class """

    modes = ["fast", "medium", "slow"]
    shortname = "SF"
    name = "scale_free"
    keywords = []
    normalize_features = True

    def compute_features(self):
        """
        Compute the scale free measures of the network

        Returns
        -------

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

        # s metric
        self.add_feature(
            "s_metric",
            lambda graph: nx.s_metric(graph, normalized=False),
            "The s-metric is defined as the sum of the products deg(u)*deg(v) for every edge (u,v) in G",
            InterpretabilityScore(4),
        )
