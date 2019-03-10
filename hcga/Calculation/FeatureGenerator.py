import numpy as np
import networkx as nx

from hcga.Operations import basic_stats
from hcga.Operations import triangles
from hcga.Operations import clustering
from hcga.Operations import centrality_degree


def FeatureGenerator(G):



    feature_dict = dict()

    feature_dict.update(triangles.triangle_stats(G))
    feature_dict.update(basic_stats.basic_stats(G))



    return feature_dict
