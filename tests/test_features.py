"""Tests suite for hcga.features."""
import pytest
import numpy as np

from hcga.extraction import get_list_feature_classes
import warnings 

warnings.simplefilter("ignore")

test_feature_classes = get_list_feature_classes(n_feats=0, mode="all", statistics_level="advanced")

def test_shortname_definition():
    """test if the class shortname has been set"""
    for feature_class in test_feature_classes:
        assert feature_class.shortname != "TP"


def test_name_definition():
    """test if the class name has been set"""
    for feature_class in test_feature_classes:
        assert feature_class.name != "template"


def test_shortnames_unique():
    """test if the shortnames are unique"""
    shortnames = [feature_class.shortname for feature_class in test_feature_classes]
    assert len(set(shortnames)) == len(shortnames)


def test_compute_features():
    """test if the features are computable"""
    from generate_test_dataset import make_test_dataset

    test_graphs, test_labels = make_test_dataset(write_to_file=False, n_graphs=1)
    for feature_class in test_feature_classes:
        for graph in test_graphs:
            print('Trying:', graph.graph['description'])
            feature_inst = feature_class(graph)
            feature_inst.compute_features()
            assert len(feature_inst.features) > 0


def test_trivial_graph():
    """test if the features are computable on trivial graph"""
    from hcga.utils import get_trivial_graph

    # TODO if they node features are zero we don't get any features
    graph = get_trivial_graph(1)
    for feature_class in test_feature_classes:
        feature_inst = feature_class(graph)
        feature_inst.compute_features()
        assert len(feature_inst.features) > 0









