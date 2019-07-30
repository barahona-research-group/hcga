# -*- coding: utf-8 -*-
"""
Created on Sun Mar  3 18:30:46 2019

@author: Rob
"""

import pandas as pd
import numpy as np
import networkx as nx

#from networkx.algorithms.community import asyn_fluidc
from networkx.algorithms.community import quality
from hcga.Operations.utils import clustering_quality

from collections import Counter
from networkx.exception import NetworkXError
from networkx.algorithms.components import is_connected
from networkx.utils import groups
from networkx.utils import not_implemented_for
from networkx.utils import py_random_state


class AsynfluidCommunities():
    def __init__(self, G):
        self.G = G
        self.feature_names = []
        self.features = {}

    def feature_extraction(self):

        """
        
        """

        # Defining the input arguments
        kmax = 10
        
        """
        feature_names = []
        """

        G = self.G

        feature_list = {}
        
        if not nx.is_directed(G):
            # basic normalisation parameters
            N = G.number_of_nodes()
            E = G.number_of_edges()

        
        
            for i in range(2,kmax):    
                
                c,density = list(asyn_fluidc(G,2))       
                
                #total density
                feature_list['total_density_'+str(i)]=sum(density)
                """
                feature_names.append('total_density_'+str(i))
                """
                
                # ratio density
                feature_list['ratio_density_'+str(i)]=np.min(density)/np.max(density)
                """
                feature_names.append('ratio_density_'+str(i))
                """

                # length of most dense community
                feature_list['most_dense_'+str(i)]=len(c[np.argmax(density)])
                """
                feature_names.append('most_dense_'+str(i))
                """
                
                # length of least dense community
                feature_list['least_dense_'+str(i)]=len(c[np.argmin(density)])
                """
                feature_names.append('least_dense_'+str(i))
                """

                # clustering quality functions       
                qual_names,qual_vals = clustering_quality(G,c)
                
                """
                # adding arguments to names
                [name+'_'+str(i) for name in qual_names]
                
                feature_list = feature_list + qual_vals
                feature_names = feature_names + qual_names    
                """
                
                for j in range(len(qual_names)):
                    feature_list[qual_names[j]+'_'+str(i)]=qual_vals[j]
                    
                    
                # calculate size ratio of the top 2 largest communities
                feature_list['num_nodes_ratio_'+str(i)]=(len(c[0])/len(c[1]))
                """
                feature_names.append('num_nodes_ratio_'+str(i))
                
                """
        else:
            qual_names = ['mod','coverage','performance','inter_comm_edge','inter_comm_nedge','intra_comm_edge']
            
            for i in range(2,kmax):
                feature_list['total_density_'+str(i)]=np.nan
                feature_list['ratio_density_'+str(i)]=np.nan
                feature_list['most_dense_'+str(i)]=np.nan
                feature_list['least_dense_'+str(i)]=np.nan
                
                for j in range(len(qual_names)):
                    feature_list[qual_names[j]+'_'+str(i)]=np.nan
                
                feature_list['num_nodes_ratio_'+str(i)]=np.nan
                
            
            
            
        """
        self.feature_names = feature_names
        """
        self.features = feature_list





@py_random_state(3)
def asyn_fluidc(G, k, max_iter=100, seed=None):
    """Returns communities in `G` as detected by Fluid Communities algorithm.

    The asynchronous fluid communities algorithm is described in
    [1]_. The algorithm is based on the simple idea of fluids interacting
    in an environment, expanding and pushing each other. It's initialization is
    random, so found communities may vary on different executions.

    The algorithm proceeds as follows. First each of the initial k communities
    is initialized in a random vertex in the graph. Then the algorithm iterates
    over all vertices in a random order, updating the community of each vertex
    based on its own community and the communities of its neighbours. This
    process is performed several times until convergence.
    At all times, each community has a total density of 1, which is equally
    distributed among the vertices it contains. If a vertex changes of
    community, vertex densities of affected communities are adjusted
    immediately. When a complete iteration over all vertices is done, such that
    no vertex changes the community it belongs to, the algorithm has converged
    and returns.

    This is the original version of the algorithm described in [1]_.
    Unfortunately, it does not support weighted graphs yet.

    Parameters
    ----------
    G : Graph

    k : integer
        The number of communities to be found.

    max_iter : integer
        The number of maximum iterations allowed. By default 15.

    seed : integer, random_state, or None (default)
        Indicator of random number generation state.
        See :ref:`Randomness<randomness>`.

    Returns
    -------
    communities : iterable
        Iterable of communities given as sets of nodes.

    Notes
    -----
    k variable is not an optional argument.

    References
    ----------
    .. [1] Parés F., Garcia-Gasulla D. et al. "Fluid Communities: A
       Competitive and Highly Scalable Community Detection Algorithm".
       [https://arxiv.org/pdf/1703.09307.pdf].
    """
    # Initial checks
    if not isinstance(k, int):
        raise NetworkXError("k must be an integer.")
    if not k > 0:
        raise NetworkXError("k must be greater than 0.")
    if not is_connected(G):
        raise NetworkXError("Fluid Communities require connected Graphs.")
    if len(G) < k:
        raise NetworkXError("k cannot be bigger than the number of nodes.")
    # Initialization
    max_density = 1.0
    vertices = list(G)
    seed.shuffle(vertices)
    communities = {n: i for i, n in enumerate(vertices[:k])}
    density = {}
    com_to_numvertices = {}
    for vertex in communities.keys():
        com_to_numvertices[communities[vertex]] = 1
        density[communities[vertex]] = max_density
    # Set up control variables and start iterating
    iter_count = 0
    cont = True
    while cont:
        cont = False
        iter_count += 1
        # Loop over all vertices in graph in a random order
        vertices = list(G)
        seed.shuffle(vertices)
        for vertex in vertices:
            # Updating rule
            com_counter = Counter()
            # Take into account self vertex community
            try:
                com_counter.update({communities[vertex]:
                                    density[communities[vertex]]})
            except KeyError:
                pass
            # Gather neighbour vertex communities
            for v in G[vertex]:
                try:
                    com_counter.update({communities[v]:
                                        density[communities[v]]})
                except KeyError:
                    continue
            # Check which is the community with highest density
            new_com = -1
            if len(com_counter.keys()) > 0:
                max_freq = max(com_counter.values())
                best_communities = [com for com, freq in com_counter.items()
                                    if (max_freq - freq) < 0.0001]
                # If actual vertex com in best communities, it is preserved
                try:
                    if communities[vertex] in best_communities:
                        new_com = communities[vertex]
                except KeyError:
                    pass
                # If vertex community changes...
                if new_com == -1:
                    # Set flag of non-convergence
                    cont = True
                    # Randomly chose a new community from candidates
                    new_com = seed.choice(best_communities)
                    # Update previous community status
                    try:
                        com_to_numvertices[communities[vertex]] -= 1
                        density[communities[vertex]] = max_density / \
                            com_to_numvertices[communities[vertex]]
                    except KeyError:
                        pass
                    # Update new community status
                    communities[vertex] = new_com
                    com_to_numvertices[communities[vertex]] += 1
                    density[communities[vertex]] = max_density / \
                        com_to_numvertices[communities[vertex]]
        # If maximum iterations reached --> output actual results
        if iter_count > max_iter:
            break
    # Return results by grouping communities as list of vertices
    return list(iter(groups(communities).values())), list(density.values())