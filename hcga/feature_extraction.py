"""functions to extract features from graphs"""
import multiprocessing
import time
from importlib import import_module
from pathlib import Path

import numpy as np


def extract(graphs, n_workers):
    """main function to extract features"""

    all_features_raw = compute_all_features(graphs, n_workers=n_workers)
    all_features, features_info_all = set_feature_ids(all_features_raw)
    feature_matrix, selected_ids = extract_feature_matrix(all_features)
    features_info = _relabel_feature_ids(features_info_all, selected_ids)

    print(
        "Feature extracted:", len(features_info_all), ", selected:", len(features_info)
    )

    return feature_matrix, features_info


def _relabel_feature_ids(features_info_all, selected_ids):
    """rname the keys in features info to match the index of the feature matrix"""
    features_info = {}
    id_counter = 0
    for feature in features_info_all:
        if feature in selected_ids:
            features_info[id_counter] = features_info_all[feature]
            id_counter += 1
    return features_info


class Worker:
    """worker for computing features"""

    def __init__(self, mode):
        self.mode = mode

    def __call__(self, graph):
        return feature_extraction(graph, self.mode)


def compute_all_features(graphs, mode="fast", n_workers=1):
    """compute the feature from all graphs"""
    worker = Worker(mode)
    if n_workers == 1:
        mapper = map
    else:
        pool = multiprocessing.Pool(n_workers)
        mapper = pool.map

    return list(mapper(worker, graphs))


def _load_feature_class(feature_name, graph=None):
    """load the feature class from feature name"""
    feature_module = import_module("hcga.features." + feature_name)
    return getattr(feature_module, feature_module.featureclass_name)(graph)


def feature_extraction(graph, mode="slow", with_runtimes=False):
    """extract features from a single graph"""
    feature_path = Path(__file__).parent / "features"
    non_feature_files = ["__init__", "feature_class"]

    if with_runtimes:
        runtimes = {}

    all_features = {}
    for f_name in feature_path.glob("*.py"):
        feature_name = f_name.stem
        if feature_name not in non_feature_files:

            if with_runtimes:
                start_time = time.time()

            feature_class = _load_feature_class(feature_name, graph)
            feature_class.update_features(all_features, mode)

            if with_runtimes:
                runtimes[feature_class.shortname] = time.time() - start_time

    if with_runtimes:
        return all_features, runtimes
    return all_features


def set_feature_ids(all_features_raw):
    """convert the raw feature to a dict by feature id, and corresponding info dict"""

    feature_path = Path(__file__).parent / "features"
    non_feature_files = ["__init__", "feature_class"]

    features_info = {}  # collect features info keyed by feature ids
    all_features = {}  # collect feature values, keyed by feature ids

    current_feature_id = 0
    for f_name in feature_path.glob("*.py"):
        feature_name = f_name.stem
        if feature_name not in non_feature_files:

            feature_class = _load_feature_class(feature_name)
            feature_class_info = feature_class.get_info()

            for feature in all_features_raw[0][feature_class.shortname]:
                #features_info[current_feature_id] = feature_class_info.copy()
                #features_info[current_feature_id]["feature"] = feature
                features_info[current_feature_id] = feature_class.get_feature_info(feature)


                all_features[current_feature_id] = []
                for i, _ in enumerate(all_features_raw):
                    all_features[current_feature_id].append(
                        all_features_raw[i][feature_class.shortname][feature]
                    )

                current_feature_id += 1

    return all_features, features_info


def extract_feature_matrix(all_features):
    """filter features and create feature matrix"""
    selected_ids = []
    for feature_id in all_features:
        # is there is a nan value
        if any(np.isnan(all_features[feature_id])):
            pass
        # is there is an inf value
        elif any(np.isinf(all_features[feature_id])):
            pass
        # if all elements are the same
        elif len(set(all_features[feature_id])) == 1:
            pass
        # else select it as a valid feature
        else:
            selected_ids.append(feature_id)

    feature_matrix = np.array([all_features[feature_id] for feature_id in selected_ids])

    return feature_matrix, selected_ids
