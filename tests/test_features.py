"""Tests suite for hcga.features."""

import warnings

import numpy as np
import pytest

from hcga.extraction import get_list_feature_classes

warnings.simplefilter("ignore")

test_feature_classes, _ = get_list_feature_classes(mode="all", statistics_level="advanced")


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
    from hcga.dataset_creation import make_test_dataset

    test_graphs = make_test_dataset(write_to_file=False, n_graphs=1)
    for feature_class in test_feature_classes:
        for graph in test_graphs:
            feature_inst = feature_class(graph)
            feature_inst.get_features()


def test_trivial_graph():
    """test if the features are computable on trivial graph"""
    from hcga.utils import get_trivial_graph

    graph = get_trivial_graph()
    for feature_class in test_feature_classes:
        feature_inst = feature_class(graph)
        feature_inst.get_features()
