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

from fa2 import ForceAtlas2
from hcga.Operations import utils
import numpy as np


class ForceCentrality():
    """
    Force centrality class
    """
    def __init__(self, G):
        self.G = G
        self.feature_names = []
        self.features = {}

    def feature_extraction(self):

        """Compute the force centrality for the graph `G`.



        Parameters
        ----------
        G : graph
          A networkx graph


        Returns
        -------
        feature_list : Dict
           Dictionary of features related to force centrality.


        Notes
        -----
        This feature uses the force atlas package to create a force-directed graph.
        
        Force atlas computed using:
            `Force Atlas <https://github.com/bhargavchippada/forceatlas2>`_

        """

        # Defining the input arguments
        bins = [10]
        

        G = self.G
        
        feature_list = {}
        
        # number of times to average force centrality
        n_force = 20
        
        #find node position with force atlas, and distance to the center is the centrality
        forceatlas2 = ForceAtlas2(
                # Tuning
                scalingRatio=2.0,
                strongGravityMode=False,
                gravity=1.0,
                # Log
                verbose=False)
        
        #producing a zero array in case force atlas fails.        
        c = np.zeros(G.number_of_nodes())

        try:
            pos = forceatlas2.forceatlas2_networkx_layout(G, pos=None, iterations=2000)
            c = np.linalg.norm(np.array(list(pos.values())),axis=1)        

        except Exception as e:
            print('Exception for centrality_force', e)
            
            
        # this changes each time so we must average of n_force times
        for i in range(n_force-1):
            try:
                pos = forceatlas2.forceatlas2_networkx_layout(G, pos=None, iterations=2000)
                c += np.linalg.norm(np.array(list(pos.values())), axis=1)                    
                c = -c/n_force

            except Exception as e:
                print('Exception for centrality_force (2nd)', e)


        feature_list['mean']=np.mean(c)
        feature_list['std']=np.std(c)
        
        feature_list['max']=np.max(abs(c))
        feature_list['min']=np.min(abs(c))

        
        for i in range(len(bins)):

            # Fitting the c distribution and finding the optimal
            # distribution according to SSE
            opt_mod,opt_mod_sse = utils.best_fit_distribution(c,bins=bins[i])
            feature_list['opt_model_{}'.format(bins[i])] = opt_mod

            # Fitting power law and finding 'a' and the SSE of fit.
            feature_list['powerlaw_a_{}'.format(bins[i])] = utils.power_law_fit(c,bins=bins[i])[0][-2]# value 'a' in power law
            feature_list['powerlaw_SSE_{}'.format(bins[i])] = utils.power_law_fit(c,bins=bins[i])[1] # value sse in power law

        

        self.features = feature_list
