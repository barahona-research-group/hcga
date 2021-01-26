""" The master template for a feature class.

Each feature class in the ./features folder can inherit the main feature class functionality.

The functions here are necessary to evaluate each individual feature found inside a feature class.

"""
import logging
import signal
import sys
import warnings

import numpy as np
import pandas as pd
import scipy.stats as st
from networkx import to_undirected
from networkx.algorithms.community import quality
from networkx.exception import NetworkXNotImplemented

from hcga.utils import TimeoutError, get_trivial_graph, timeout_handler

L = logging.getLogger(__name__)
warnings.simplefilter("ignore")

import dill


def run_dill_encoded(payload, q_worker):
    fun, args = dill.loads(payload)
    return fun(*args, q_worker=q_worker)


def _lemmiwinks(func, args, kwargs, q):
    q.put(func(*args, **kwargs))


class FeatureClass:
    """ Main functionality to be inherited by each feature class"""

    # Class variables that describe the feature,
    # They should be defined for all child features
    modes = ["fast", "medium", "slow"]
    shortname = "TP"
    name = "template"

    statistics_level = "basic"
    encoding = None
    normalize_features = True
    with_node_features = False
    timeout = 10

    # Feature descriptions as class variable
    feature_descriptions = {}

    @classmethod
    def _get_doc(cls):
        f_info = cls.setup_class()
        doc_string = "\n This class produces the following features:\n"
        _indent = "    "
        for feature in f_info.columns:
            if f_info.loc["shortname", feature] == cls.shortname:
                doc_string += "Attributes: \n"
                doc_string += _indent + feature + ": \n"
                doc_string += 2 * _indent + "Args: \n"
                for data in f_info[feature].index:
                    doc_string += (
                        3 * _indent + str(data) + ": " + str(f_info.loc[data, feature]) + "\n"
                    )
        return doc_string

    def __init_subclass__(cls):
        """Initialise class variables to default for each child class."""
        cls.feature_descriptions = {}
        cls.__doc__ += cls._get_doc()

    def __init__(self, graph=None):
        """Initialise a feature class.

        Args:
            graph (Graph): graph for initialisation, converted to given encoding
        """
        if graph is not None:
            self.graph = graph.get_graph(self.__class__.encoding)
            self.graph_id = graph.id
            self.graph_type = graph.graph_type
        else:
            self.graph = None
        self.features = {}

    @classmethod
    def setup_class(
        cls,
        normalize_features=True,
        statistics_level="basic",
        n_node_features=0,
        timeout=10,
    ):
        """Initializes the class by adding descriptions for all features.

        Args:
            normalize_features (bool): normalise features by number of nodes and number of edges
            statistics_level (str): 'basic', 'advanced' - for features that provide distributions
                we must compute statistics.
            n_node_features (int):  dimension of node features for feature constructors
            timeout (int): number of seconds before the calculation for a feature is cancelled

        Returns:
            (DataFrame): dataframe with feature information
        """
        cls.normalize_features = normalize_features
        cls.statistics_level = statistics_level
        cls.n_node_features = n_node_features
        cls.timeout = timeout

        inst = cls(get_trivial_graph(n_node_features=n_node_features))
        features = inst.get_features(all_features=True)
        feature_info = pd.DataFrame()
        for feature in features:
            feat_info = inst.get_feature_info(feature)
            feature_info[feature] = pd.Series(feat_info)

        return feature_info

    def get_info(self):
        """Return a dictionary of informations about the feature class."""
        return {
            "name": self.__class__.name,
            "shortname": self.__class__.shortname,
        }

    def _test_feature_exists(self, feature_name):
        """Test if feature feature_name exists in description list."""
        if feature_name not in self.__class__.feature_descriptions:
            raise Exception("Feature {} does not exist in class {}".format(feature_name, self.name))

    def get_feature_info(self, feature_name):
        """Returns a dictionary of information about the feature feature_name."""
        self._test_feature_exists(feature_name)
        feature_dict = self.__class__.feature_descriptions[feature_name]
        feature_info = self.get_info()

        feature_info["name"] = feature_name
        feature_info["fullname"] = feature_info["shortname"] + "_" + feature_name
        feature_info["description"] = feature_dict["desc"]
        feature_info["interpretability"] = feature_dict["interpret"]
        return feature_info

    def get_feature_description(self, feature_name):
        """Returns interpretability score of the feature feature_name."""
        self._test_feature_exists(feature_name)
        feat_dict = self.__class__.feature_descriptions[feature_name]
        return feat_dict["desc"]

    def get_feature_interpretability(self, feature_name):
        """Returns interpretability score of the feature feature_name."""
        self._test_feature_exists(feature_name)
        feat_dict = self.__class__.feature_descriptions[feature_name]
        return feat_dict["interpret"]

    @classmethod
    def add_feature_description(cls, feature_name, feature_desc, feature_interpret):
        """Adds the description to the class variable if not already there."""
        if feature_name not in cls.feature_descriptions:
            cls.feature_descriptions[feature_name] = {
                "desc": feature_desc,
                "interpret": feature_interpret,
            }

    def evaluate_feature(  # pylint: disable=too-many-branches
        self,
        feature_function,
        feature_name,
        function_args=None,
        statistics=None,
    ):
        """Evaluating a feature function.

        We catch any error during a computation, which may result in Nan feature value.
        In addition, the evaluation has to be done before timoeut, or it will return Nan.

        Args:
            feature_function (function): function to evaluate to compute a feature
            feature_name (str): name of the feature
            function_args (list): additional arguments to pass to feature_function
            statistics (str): type of statistics to apply to high dimensional features.
        """
        if not callable(feature_function):
            raise Exception("The feature function {} is not callable!".format(feature_name))

        if function_args is None:
            function_args = self.graph
        import multiprocessing as mp
        import multiprocessing.queues as mpq

        q_worker = mp.Queue()
        payload = dill.dumps((_lemmiwinks, (feature_function, function_args, {})))
        proc = mp.Process(target=run_dill_encoded, args=(payload, q_worker))
        proc.start()
        try:
            try:
                feature = q_worker.get(timeout=1000.0)
            except NetworkXNotImplemented:
                if self.graph_type == "directed":
                    feature = feature_function(to_undirected(function_args))
            signal.alarm(0)
            return feature

        except mpq.Empty:
            proc.terminate()
            print("Timeout!")
            return None
        except (KeyboardInterrupt, SystemExit):
            sys.exit(0)

        except TimeoutError:
            L.debug(
                "Feature %s for graph %d took longer than %s seconds",
                feature_name,
                self.graph_id,
                str(self.timeout),
            )
            return None
        except Exception as exc:  # pylint: disable=broad-except
            if self.graph_id != -1:
                L.debug(
                    "Failed feature %s for graph %d with exception: %s",
                    feature_name,
                    self.graph_id,
                    str(exc),
                )
            return None
        if statistics == "clustering":
            if not isinstance(feature, list):
                raise Exception(
                    "Feature {} with clustering statistics is not a list: {}".format(
                        feature_name, feature
                    )
                )
            if not isinstance(feature[0], set):
                raise Exception(
                    "Feature {} with clustering statistics is not a list of sets: {}".format(
                        feature_name, feature
                    )
                )
        elif statistics in ("centrality", "node_features"):
            expected_types = (list, np.ndarray)
            if not isinstance(feature, expected_types):
                raise Exception(
                    "Feature {} with statistics does not return expected type{}: {}".format(
                        feature_name, expected_types, feature
                    )
                )
        else:
            expected_types = (int, float, np.int32, np.int64, np.float32, np.float64)
            if not isinstance(feature, expected_types):
                raise Exception(
                    "Feature {} of type {} with no stat does not return expected type{}: {}".format(
                        feature_name,
                        type(feature),
                        expected_types,
                        feature,
                    )
                )

    def add_feature(  # pylint: disable=inconsistent-return-statements
        self,
        feature_name,
        feature_function,
        feature_description,
        feature_interpret,
        function_args=None,
        statistics=None,
    ):
        """Adds a computed feature value and its description.

        Args:
            feature_name (str): name of the feature
            feature_function (function): function to evaluate to compute a feature
            feature_description (str): short description of the feature
            feature_interpret (int): interpretability score of thee feature
            function_args (list): additional arguments to pass to feature_function
            statistics (str): type of statistics to apply to high dimensional features.
        """
        func_result = self.evaluate_feature(
            feature_function,
            feature_name,
            function_args=function_args,
            statistics=statistics,
        )
        if func_result is None and not self.all_features:
            return 0

        if statistics is None:
            self.features[feature_name] = func_result
            if not hasattr(feature_interpret, "get_score"):
                feature_interpret = InterpretabilityScore(feature_interpret)

            self.__class__.add_feature_description(
                feature_name, feature_description, feature_interpret
            )

        elif statistics == "centrality":
            self._feature_statistics(
                func_result,
                feature_name,
                feature_description,
                feature_interpret,
            )

        elif statistics == "clustering":
            self._clustering_statistics(
                func_result,
                feature_name,
                feature_description,
                feature_interpret,
            )

        elif statistics == "node_features":
            self._node_feature_statistics(
                func_result,
                feature_name,
                feature_description,
                feature_interpret,
            )

        elif statistics == "list":
            self._list_statistics(
                func_result,
                feature_name,
                feature_description,
                feature_interpret,
            )

    def compute_features(self):
        """Main feature extraction function.

        This function should be used by each specific feature class to add new features.
        """
        self.add_feature("test", lambda graph: 0.0, "Test feature for the base feature class", 5)

    def get_features(self, all_features=False):
        """Compute all the possible features."""
        if self.__class__.shortname == "TP" and self.__class__.__name__ != "FeatureClass":
            raise Exception(
                "Shortname not set for feature class {}".format(self.__class__.__name__)
            )
        self.all_features = all_features
        self.compute_features()
        if self.normalize_features:
            self.compute_normalize_features()
        return self.features

    def compute_normalize_features(self):
        """Triple the number of features by normalising by node and edges."""
        interpretability_downgrade = 1
        for feature_name in list(self.features.keys()):
            feat_interpret = self.get_feature_interpretability(feature_name)
            feat_desc = self.get_feature_description(feature_name)
            self.add_feature(
                feature_name + "_N",
                lambda graph: self.features[feature_name] / len(graph.nodes),
                feat_desc + ", normalised by the number of nodes in the graph",
                feat_interpret - interpretability_downgrade,
            )
            self.add_feature(
                feature_name + "_E",
                lambda graph: self.features[feature_name] / len(graph.edges),
                feat_desc + ", normalised by the number of edges in the graph",
                feat_interpret - interpretability_downgrade,
            )

    def _clustering_statistics(self, community_partition, feat_name, feat_desc, feat_interpret):
        """Compute quality of the community partitions."""
        compl_desc = " of the partition of " + feat_desc

        self.add_feature(
            feat_name + "_modularity",
            lambda graph: quality.modularity(graph, community_partition),
            "Modularity" + compl_desc,
            feat_interpret,
        )

        self.add_feature(
            feat_name + "_coverage",
            lambda graph: quality.coverage(graph, community_partition),
            "Coverage" + compl_desc,
            feat_interpret,
        )
        self.add_feature(
            feat_name + "_performance",
            lambda graph: quality.performance(graph, community_partition),
            "Performance" + compl_desc,
            feat_interpret,
        )
        self.add_feature(
            feat_name + "_inter_community_edges",
            lambda graph: quality.inter_community_edges(graph, community_partition),
            "Inter community edges" + compl_desc,
            feat_interpret,
        )
        self.add_feature(
            feat_name + "_inter_community_non_edges",
            lambda graph: quality.inter_community_non_edges(graph, community_partition),
            "Inter community non edges" + compl_desc,
            feat_interpret,
        )
        self.add_feature(
            feat_name + "_intra_community_edges",
            lambda graph: quality.intra_community_edges(graph, community_partition),
            "Intra community edges" + compl_desc,
            feat_interpret,
        )

    def _node_feature_statistics(self, feat_dist, feat_name, feat_desc, feat_interpret):
        """Computes summary statistics of each feature distribution."""
        for node_feats in range(feat_dist.shape[1]):
            self._feature_statistics_basic(
                feat_dist[:, node_feats],
                feat_name + str(node_feats),
                feat_desc + str(node_feats),
                feat_interpret,
            )

    def _feature_statistics(self, feat_dist, feat_name, feat_desc, feat_interpret):
        """Computes summary statistics of distributions."""
        self._feature_statistics_basic(feat_dist, feat_name, feat_desc, feat_interpret)

        if self.statistics_level == "medium":
            self._feature_statistics_medium(feat_dist, feat_name, feat_desc, feat_interpret)

        if self.statistics_level == "advanced":
            self._feature_statistics_medium(feat_dist, feat_name, feat_desc, feat_interpret)
            self._feature_statistics_advanced(feat_dist, feat_name, feat_desc, feat_interpret)

    def _list_statistics(self, feat_dist, feat_name, feat_desc, feat_interpret):
        """Compute list statisttics."""
        if feat_dist is not None:
            for i in range(len(feat_dist)):
                self.add_feature(
                    feat_name[i],
                    lambda args: args[i],
                    feat_desc,
                    feat_interpret,
                    function_args=feat_dist,
                )

    def _feature_statistics_basic(self, feat_dist, feat_name, feat_desc, feat_interpret):
        """Computes basic summary statistics of distributions."""
        compl_desc = " of the distribution of " + feat_desc

        self.add_feature(
            feat_name + "_mean",
            np.mean,
            "Mean" + compl_desc,
            feat_interpret,
            function_args=feat_dist,
        )
        self.add_feature(
            feat_name + "_min",
            np.min,
            "Minimum" + compl_desc,
            feat_interpret,
            function_args=feat_dist,
        )
        self.add_feature(
            feat_name + "_sum",
            np.sum,
            "Sum" + compl_desc,
            feat_interpret,
            function_args=feat_dist,
        )
        self.add_feature(
            feat_name + "_max",
            np.max,
            "Maximum" + compl_desc,
            feat_interpret,
            function_args=feat_dist,
        )
        self.add_feature(
            feat_name + "_median",
            np.median,
            "Median" + compl_desc,
            feat_interpret,
            function_args=feat_dist,
        )
        self.add_feature(
            feat_name + "_std",
            np.std,
            "Standard deviation" + compl_desc,
            feat_interpret,
            function_args=feat_dist,
        )

    def _feature_statistics_medium(self, feat_dist, feat_name, feat_desc, feat_interpret):
        """Computes medium summary statistics of distributions."""
        compl_desc = " of the distribution of " + feat_desc

        self.add_feature(
            feat_name + "_gmean",
            st.gmean,
            "G. Mean" + compl_desc,
            feat_interpret - 1,
            function_args=feat_dist,
        )
        self.add_feature(
            feat_name + "_hmean",
            lambda dist: st.hmean(np.abs(dist) + 1e-8),
            "G. Mean" + compl_desc,
            feat_interpret - 1,
            function_args=feat_dist,
        )
        self.add_feature(
            feat_name + "_kurtosis",
            st.kurtosis,
            "Kurtosis" + compl_desc,
            feat_interpret - 1,
            function_args=feat_dist,
        )
        self.add_feature(
            feat_name + "_mode",
            lambda dist: st.mode(dist)[0][0],
            "Mode" + compl_desc,
            feat_interpret - 1,
            function_args=feat_dist,
        )

    def _feature_statistics_advanced(self, feat_dist, feat_name, feat_desc, feat_interpret):
        """Computes advanced summary statistics of distributions."""
        compl_desc = " of the distribution of " + feat_desc

        self.add_feature(
            feat_name + "_kstat",
            st.kstat,
            "Kstat" + compl_desc,
            feat_interpret - 1,
            function_args=feat_dist,
        )
        self.add_feature(
            feat_name + "_kstatvar",
            st.kstatvar,
            "Kstat variance" + compl_desc,
            feat_interpret - 1,
            function_args=feat_dist,
        )
        self.add_feature(
            feat_name + "_tmean",
            st.tmean,
            "Tmean" + compl_desc,
            feat_interpret - 1,
            function_args=feat_dist,
        )
        self.add_feature(
            feat_name + "_tmean",
            st.tvar,
            "Tvariance" + compl_desc,
            feat_interpret - 1,
            function_args=feat_dist,
        )
        self.add_feature(
            feat_name + "_tmin",
            st.tmin,
            "Tminimum" + compl_desc,
            feat_interpret - 1,
            function_args=feat_dist,
        )
        self.add_feature(
            feat_name + "_tmax",
            st.tmax,
            "Tmaximum" + compl_desc,
            feat_interpret - 1,
            function_args=feat_dist,
        )
        self.add_feature(
            feat_name + "_tstd",
            st.tstd,
            "T standard deviation" + compl_desc,
            feat_interpret - 1,
            function_args=feat_dist,
        )
        self.add_feature(
            feat_name + "_tsem",
            st.tsem,
            "T sem" + compl_desc,
            feat_interpret - 1,
            function_args=feat_dist,
        )
        self.add_feature(
            feat_name + "_variation",
            st.variation,
            "Variation" + compl_desc,
            feat_interpret - 1,
            function_args=feat_dist,
        )
        self.add_feature(
            feat_name + "_mean_repeats",
            lambda dist: np.mean(st.find_repeats(dist)[1]),
            "Mean repeats" + compl_desc,
            feat_interpret - 1,
            function_args=feat_dist,
        )
        self.add_feature(
            feat_name + "_entropy",
            st.entropy,
            "Entropy" + compl_desc,
            feat_interpret - 1,
            function_args=feat_dist,
        )
        self.add_feature(
            feat_name + "_sem",
            st.sem,
            "SEM" + compl_desc,
            feat_interpret - 1,
            function_args=feat_dist,
        )


class InterpretabilityScore:
    """Class to represent interpretability scores of features."""

    min_score = 0
    max_score = 5

    def __init__(self, score):
        """Init function for InterpretabilityScore.

        Args:
            score: (int/{'min', 'max'} value of score to set
        """
        if score == "max":
            score = self.max_score
        elif score == "min":
            score = self.min_score
        self.set_score(score)

    def set_score(self, score):
        """Set the interpretability score."""
        score_int = int(score)
        if score_int < self.min_score:
            score_int = self.min_score
        elif score_int > self.max_score:
            score_int = self.max_score
        self.score = score_int

    def get_score(self):
        """Get the interpretability score."""
        return self.score

    def __add__(self, other):
        if isinstance(other, int):
            othervalue = other
        else:
            othervalue = other.get_score()
        result = self.__class__(self.get_score() + othervalue)
        return result

    def __sub__(self, other):
        if isinstance(other, int):
            othervalue = other
        else:
            othervalue = other.get_score()
        result = self.__class__(self.get_score() - othervalue)
        return result

    def __str__(self):
        return str(self.score)

    def __repr__(self):
        return self.__class__.__name__ + "({})".format(self.__str__())
