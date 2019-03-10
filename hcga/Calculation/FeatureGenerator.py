import numpy as np
import networkx as nx

from hcga.Operations import basic_stats
from hcga.Operations import triangles
from hcga.Operations import heuristics


def FeatureGenerator(G):

    feature_dict = dict()

    feature_dict.update(triangles.triangle_stats(G))
    feature_dict.update(basic_stats.basic_stats(G))



    return feature_dict
