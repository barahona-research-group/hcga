#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  7 10:41:52 2019

@author: robert
"""

from networkx.algorithms import centrality
from hcga.Operations import utils
import numpy as np
import scipy as sp


class EigenCentrality():
    """
    Centrality eigenvector
    """
    def __init__(self, G, eigens):
        self.G = G
        self.eigenvectors = eigens[1]
        self.eigenvalues = eigens[0]
        
        self.feature_names = []
        self.features = []

    def feature_extraction(self):

        """Compute the eigenvector centrality for the graph `G`.

        Eigenvector centrality computes the centrality for a node based on the
        centrality of its neighbors. The eigenvector centrality for node $i$ is
        the $i$-th element of the vector $x$ defined by the equation

        .. math::

            Ax = \lambda x

        where $A$ is the adjacency matrix of the graph `G` with eigenvalue
        $\lambda$. By virtue of the Perron–Frobenius theorem, there is a unique
        solution $x$, all of whose entries are positive, if $\lambda$ is the
        largest eigenvalue of the adjacency matrix $A$ ([2]_).

        Parameters
        ----------
        G : graph
          A networkx graph

        args: list
            Parameters for calculating feature_list
                arg[0]: integer
                    number of bins


        Returns
        -------
        feature_list :list
           List of features related to eigenvector centrality.


        Notes
        -----

        Implemented using networkx:
            https://networkx.github.io/documentation/stable/_modules/networkx/algorithms/centrality/eigenvector.html#eigenvector_centrality

        The measure was introduced by [1]_ and is discussed in [2]_.

        References
        ----------
        .. [1] Phillip Bonacich.
           "Power and Centrality: A Family of Measures."
           *American Journal of Sociology* 92(5):1170–1182, 1986
           <http://www.leonidzhukov.net/hse/2014/socialnetworks/papers/Bonacich-Centrality.pdf>
        .. [2] Mark E. J. Newman.
           *Networks: An Introduction.*
           Oxford University Press, USA, 2010, pp. 169.

        """

        # Defining the input arguments
        bins = [10,20,50]
        
        """
        # Defining featurenames
        feature_names = ['mean','std','max','min']
        """

        #G = self.G
        #op_obj = self.op_obj

        feature_list = {}

        eigenvalues = self.eigenvalues
        
        
        # extract the precomputed eigenvectors from the operations object
        eigenvector = self.eigenvectors[:,np.argmax(eigenvalues.real)]
            
        #eigenvector = eigenvectors[:,0]
        
        largest = eigenvector.flatten().real
        norm = sp.sign(largest.sum()) * sp.linalg.norm(largest)
        eigenvector_centrality = largest / norm

        # Basic stats regarding the eigenvector centrality distribution
        feature_list['mean'] = eigenvector_centrality.mean()
        feature_list['std'] = eigenvector_centrality.std()
        feature_list['max'] = eigenvector_centrality.max()
        feature_list['min'] = eigenvector_centrality.min()
        
        for i in range(len(bins)):
            """# Adding to feature names
            feature_names.append('opt_model_{}'.format(bins[i]))
            feature_names.append('powerlaw_a_{}'.format(bins[i]))
            feature_names.append('powerlaw_SSE_{}'.format(bins[i]))"""
            
            # Fitting the eigenvector centrality distribution and finding the optimal
            # distribution according to SSE
            opt_mod,opt_mod_sse = utils.best_fit_distribution(eigenvector_centrality,bins=bins[i])
            feature_list['opt_model_{}'.format(bins[i])] = opt_mod

            # Fitting power law and finding 'a' and the SSE of fit.
            feature_list['powerlaw_a_{}'.format(bins[i])] = utils.power_law_fit(eigenvector_centrality,bins=bins[i])[0][-2]# value 'a' in power law
            feature_list['powerlaw_SSE_{}'.format(bins[i])] = utils.power_law_fit(eigenvector_centrality,bins=bins[i])[1] # value sse in power law

        #Calculate the degree centrality of each node
#        try:
#            eigenvector_centrality = np.asarray(list(centrality.eigenvector_centrality(G).values()))
#
#            # Basic stats regarding the degree centrality distribution
#            feature_list.append(eigenvector_centrality.mean())
#            feature_list.append(eigenvector_centrality.std())
#
#            # Fitting the degree centrality distribution and finding the optimal
#            # distribution according to SSE
#            opt_mod,opt_mod_sse = utils.best_fit_distribution(eigenvector_centrality,bins=bins)
#            feature_list.append(opt_mod)
#
#            # Fitting power law and finding 'a' and the SSE of fit.
#            feature_list.append(utils.power_law_fit(eigenvector_centrality,bins=bins)[0][-2]) # value 'a' in power law
#            feature_list.append(utils.power_law_fit(eigenvector_centrality,bins=bins)[1]) # value sse in power law
#        except:
#            feature_list = np.empty(len(self.feature_names))
#            feature_list.fill(np.nan)
#            feature_list = feature_list.tolist()
#            pass



        # Fitting normal distribution and finding...


        # Fitting exponential and finding ...

        """
        self.feature_names=feature_names
        """
        self.features = feature_list
