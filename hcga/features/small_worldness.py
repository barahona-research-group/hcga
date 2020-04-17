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

featureclass_name = "SmallWorldness"


class SmallWorldness(FeatureClass):
    """ Small worldness class """

    modes = ["slow"]
    shortname = "SW"
    name = "small_worldness"
    keywords = []
    normalize_features = True

    def compute_features(self):
        """
        Compute the small world measures of the network

        The small-world coefficient of a graph G is:
    
        omega = Lr/L - C/Cl
    
        where C and L are respectively the average clustering coefficient and
        average shortest path length of G. Lr is the average shortest path length
        of an equivalent random graph and Cl is the average clustering coefficient
        of an equivalent lattice graph.
    
        The small-world coefficient (omega) ranges between -1 and 1. Values close
        to 0 means the G features small-world characteristics. Values close to -1
        means G has a lattice shape whereas values close to 1 means G is a random
        graph.
        
        Parameters
        ----------
        G : graph
           A networkx graph

        Returns
        -------

        Notes
        -----
        Small world calculations using networkx:
            `Networkx_omega <https://networkx.github.io/documentation/latest/_modules/networkx/algorithms/smallworld.html#omega>`_
        
        The implementation is adapted from the algorithm by Telesford et al. [1]_.
    
        References
        ----------
        .. [1] Telesford, Joyce, Hayasaka, Burdette, and Laurienti (2011).
               "The Ubiquity of Small-World Networks".
               Brain Connectivity. 1 (0038): 367-75.  
               
        """

        # omega metric
        self.add_feature(
            "omega",
            lambda graph: nx.omega(graph),
            "The small world coefficient omega",
            InterpretabilityScore(4),
        )


