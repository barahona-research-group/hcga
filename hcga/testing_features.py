"""
Testing suite for hcga.features

Run all tests with the command 'python testing_features.py'
"""

import unittest
import numpy as np

from hcga.make_data import make_test_data
from hcga.feature_extraction import _get_list_feature_classes

test_graphs, test_labels = make_test_data(save_data=False)
test_feature_classes = _get_list_feature_classes(mode="all")


class TestFeatureClasses(unittest.TestCase):
    def test_shortname_definition(self):
        for feature_class in test_feature_classes:
            with self.subTest(feature_class=feature_class.__name__,):
                self.assertNotEqual(feature_class.shortname, "TP")

    def test_name_definition(self):
        for feature_class in test_feature_classes:
            with self.subTest(feature_class=feature_class.__name__,):
                self.assertNotEqual(feature_class.name, "template")

    def test_shortnames_unique(self):
        shortnames = []
        for feature_class in test_feature_classes:
            shortnames.append(feature_class.shortname)
        shortnames = sorted(shortnames)
        unique_shortnames = sorted(list(set(shortnames)))
        self.assertListEqual(shortnames, unique_shortnames)

    def test_compute_features(self):
        for feature_class in test_feature_classes:
            for graph in test_graphs:
                with self.subTest(
                    feature_class=feature_class.__name__,
                    graph=graph.graph["description"],
                ):
                    feature_inst = feature_class(graph)
                    feature_inst.compute_features()
                    self.assertTrue(len(feature_inst.features) > 0)


class TestIndividualFeatures(unittest.TestCase):
    def test_features_not_nan(self):
        for feature_class in test_feature_classes:
            for graph in test_graphs:
                feature_inst = feature_class(graph)
                feature_inst.compute_features()
                # self.assertFalse(np.nan in feature_inst.features.values())
                for feat, val in feature_inst.features.items():
                    with self.subTest(
                        feature_class=feature_class.__name__,
                        feature=feat,
                        graph=graph.graph["description"],
                    ):
                        self.assertTrue(np.isfinite(val))


if __name__ == "__main__":
    unittest.main()
