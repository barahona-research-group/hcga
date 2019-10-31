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


class SpectrumModularity():
    """
    Spectrum modularity class
    """
    def __init__(self, G):
        self.G = G
        self.feature_names = []
        self.features = {}

    def feature_extraction(self):

        """Compute measures of the spectrum of the modularity matrix



        Parameters
        ----------
        G : graph
          A networkx graph




        Returns
        -------
        feature_list : dict
           Dictionary of features related to spectrum of the modularity matrix


        Notes
        -----
        Spectral modularity calculations using networkx:
            `Networkx_spectrum_modularity <https://networkx.github.io/documentation/stable/reference/generated/networkx.linalg.spectrum.modularity_spectrum.html>`_


        """
        
        
        G = self.G

        feature_list = {}         
        

        # laplacian spectrum
        eigenvals_M = np.real(nx.linalg.spectrum.modularity_spectrum(G))
        
        if len(eigenvals_M) < 10:
            eigenvals_M = np.concatenate((eigenvals_M,np.zeros(10-len(eigenvals_M))))        
            
        for i in range(10):               
            feature_list['M_eigvals_'+str(i)]=eigenvals_M[i]
        
        
        for i in range(10):
            for j in range(i):
                if i != j:
                    try:
                        if abs(eigenvals_M[j] ) < 1e-8:
                            feature_list['M_eigvals_ratio_'+str(i)+'_'+str(j)] = 0 
                        else:
                            feature_list['M_eigvals_ratio_'+str(i)+'_'+str(j)] = eigenvals_M[i]/eigenvals_M[j]
                    except Exception as e:
                        print('Exception for spectrum_modularity', e)
                        feature_list['M_eigvals_ratio_'+str(i)+'_'+str(j)] = np.nan
                
        feature_list['M_eigvals_min'] = min(eigenvals_M)




        

        self.features = feature_list
