"""functions to extract features from graphs"""
import multiprocessing
import time
from importlib import import_module
from pathlib import Path
import networkx as nx
import numpy as np


def extract(graphs, n_workers, mode='fast'):
    """main function to extract features"""

    feat_classes = _get_list_feature_classes(mode)

    all_features_raw = compute_all_features(graphs, feat_classes, n_workers=n_workers)
    all_features, features_info_all = set_feature_ids(all_features_raw, feat_classes)
    feature_matrix, selected_ids = extract_feature_matrix(all_features)
    features_info = _relabel_feature_ids(features_info_all, selected_ids)

    print(
        "Feature extracted:", len(features_info_all), ", selected:", len(features_info)
    )

    return feature_matrix, features_info

def _get_list_feature_classes(mode='fast'):
    """Generates and returns the list of feature classes to compute for a given mode"""
    feature_path = Path(__file__).parent / "features"
    non_feature_files = ["__init__", "feature_class"]

    list_feature_classes = []
    trivial_graph = nx.generators.classic.complete_graph(3)

    for f_name in feature_path.glob("*.py"):
        feature_name = f_name.stem
        if feature_name not in non_feature_files:
            feature_class = _load_feature_class(feature_name)
            if mode in feature_class.modes:
                list_feature_classes.append(feature_class)
                # runs once update_feature with trivial graph to create class variables
                feature_class(trivial_graph).update_features({})
    return list_feature_classes


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

    def __init__(self, list_feature_classes):
        self.list_feature_classes = list_feature_classes

    def __call__(self, graph):
        return feature_extraction(graph, self.list_feature_classes)


def compute_all_features(graphs, list_feature_classes, n_workers=1):
    """compute the feature from all graphs"""
    worker = Worker(list_feature_classes)
    if n_workers == 1:
        mapper = map
    else:
        pool = multiprocessing.Pool(n_workers)
        mapper = pool.map

    return list(mapper(worker, graphs))


def _load_feature_class(feature_name):
    """load the feature class from feature name"""
    feature_module = import_module("hcga.features." + feature_name)
    return getattr(feature_module, feature_module.featureclass_name)


def feature_extraction(graph, list_feature_classes, with_runtimes=False):
    """extract features from a single graph"""

    if with_runtimes:
        runtimes = {}

    all_features = {}
    for feature_class in list_feature_classes:
        if with_runtimes:
            start_time = time.time()

        feature_inst = feature_class(graph)
        feature_inst.update_features(all_features)

        if with_runtimes:
            runtimes[feature_class.shortname] = time.time() - start_time

    if with_runtimes:
        return all_features, runtimes
    return all_features


def set_feature_ids(all_features_raw, list_feature_classes):
    """convert the raw feature to a dict by feature id, and corresponding info dict"""

    features_info = {}  # collect features info keyed by feature ids
    all_features = {}  # collect feature values, keyed by feature ids

    current_feature_id = 0

    for feature_class in list_feature_classes:
        feature_inst = feature_class()

        for feature in all_features_raw[0][feature_inst.shortname]:
            features_info[current_feature_id] = feature_inst.get_feature_info(feature)

            all_features[current_feature_id] = []
            for i, _ in enumerate(all_features_raw):
                all_features[current_feature_id].append(
                    all_features_raw[i][feature_inst.shortname][feature]
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

