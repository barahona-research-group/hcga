"""
Testing suite for hcga.features

Run all tests with the command 'python testing_features.py'
"""

import unittest
import numpy as np

from hcga.make_data import make_test_data
from hcga.feature_extraction import _get_list_feature_classes

test_graphs, test_labels = make_test_data(save_data=False)
test_feature_classes = _get_list_feature_classes(mode="fast")

class TestingFeatureClasses(unittest.TestCase):

    def test_features_not_nan(self):
        for feature_class in test_feature_classes:
            for graph in test_graphs:
                feature_inst = feature_class(graph)
                feature_inst.compute_features()
                #self.assertFalse(np.nan in feature_inst.features.values())
                for feat, val in feature_inst.features.items():
                    with self.subTest(
                            feature_class=feature_class.__name__, 
                            feature=feat,
                            graph=graph.graph['description'],
                            ):
                        self.assertFalse(np.isnan(val))


if __name__ == '__main__':
    unittest.main()

