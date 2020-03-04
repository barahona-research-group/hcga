"""utils functions"""
import numpy as np


def filter_features(features):
    """filter features and create feature matrix"""

    # remove inf and nan
    nan_features = features.replace([np.inf, -np.inf], np.nan)
    valid_features = nan_features.dropna(axis=1)

    # remove features with equal values accros graphs
    return valid_features.drop(
        valid_features.std()[(valid_features.std() == 0)].index, axis=1
    )
