from networkx.algorithms import centrality
from hcga.Operations import utils
import numpy as np



class EigenCentrality():
    def __init__(self, G):
        self.G = G
        self.feature_names = []
        self.features = []

    def feature_extraction(self,args):

        r"""Compute the eigenvector centrality for the graph `G`.

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
        bins = args[0]

        # Defining featurenames
        self.feature_names = ['mean','std','opt_model','powerlaw_a','powerlaw_SSE']

        G = self.G

        feature_list = []

        #Calculate the degree centrality of each node
        try:
            eigenvector_centrality = np.asarray(list(centrality.eigenvector_centrality(G).values()))

            # Basic stats regarding the degree centrality distribution
            feature_list.append(eigenvector_centrality.mean())
            feature_list.append(eigenvector_centrality.std())

            # Fitting the degree centrality distribution and finding the optimal
            # distribution according to SSE
            opt_mod,opt_mod_sse = utils.best_fit_distribution(eigenvector_centrality,bins=bins)
            feature_list.append(opt_mod)

            # Fitting power law and finding 'a' and the SSE of fit.
            feature_list.append(utils.power_law_fit(eigenvector_centrality,bins=bins)[0][-2]) # value 'a' in power law
            feature_list.append(utils.power_law_fit(eigenvector_centrality,bins=bins)[1]) # value sse in power law
        except:
            feature_list = np.empty(len(self.feature_names))
            feature_list.fill(np.nan)
            feature_list = feature_list.tolist()
            pass



        # Fitting normal distribution and finding...


        # Fitting exponential and finding ...


        self.features = feature_list
