"""template class for feature extraction"""
import logging
import sys
from functools import partial


import networkx as nx
import numpy as np
import scipy.stats as st
from networkx.algorithms.community import quality

from . import utils

logging.basicConfig(
    filename="feature_exceptions.log", filemode="w", level=logging.DEBUG
)
L = logging.getLogger("Feature exceptions")


class FeatureClass:
    """template class"""

    # Class variables that describe the feature,
    # They should be defined for all child features
    modes = ["fast", "medium", "slow"]
    shortname = "TP"
    name = "template"
    keywords = ["template"]

    statistics_level = "basic"
    normalize_features = True

    # Feature descriptions as class variable
    feature_descriptions = {}

    trivial_graph = utils.get_trivial_graph()

    def __init_subclass__(cls):
        """Initialise class variables to default for each child class"""
        cls.feature_descriptions = {}

    def __init__(self, graph=None):
        """init function"""
        self.graph = graph
        if graph is not None:
            self.verify_graph()
        self.features = {}

    def verify_graph(self):
        """make sure a graph has correct properties"""
        if nx.is_directed(self.graph):
            self.graph = self.graph.to_undirected()

        if "id" not in self.graph.graph:
            L.warning("An id has not been set for a graph")
            self.graph.graph["id"] = -1

    @classmethod
    def setup_class(cls, normalize_features=True, statistics_level="basic"):
        """Initializes the class by adding descriptions for all features"""
        cls.normalize_features = normalize_features
        cls.statistics_level = statistics_level

        # runs once update_feature on None graph to populate feature descriptions
        inst = cls(cls.trivial_graph)
        inst.update_features({})

    def get_info(self):
        """return a dictionary of informations about the feature class"""
        return {
            "name": self.__class__.name,
            "shortname": self.__class__.shortname,
            "keywords": self.__class__.keywords,
        }

    def _test_feature_exists(self, f_name):
        """Test if feature f_name exists in description list"""
        if f_name not in self.__class__.feature_descriptions:
            raise Exception(
                "Feature {} does not exist in class {}".format(f_name, self.name)
            )

    def get_feature_info(self, f_name):
        """Returns a dictionary of information about the feature f_name"""
        self._test_feature_exists(f_name)
        feat_info = self.get_info()
        feat_info["feature_name"] = f_name
        feat_dict = self.__class__.feature_descriptions[f_name]
        feat_info["feature_description"] = feat_dict["desc"]
        feat_info["feature_interpretability"] = feat_dict["interpret"]
        return feat_info

    def get_feature_description(self, f_name):
        """Returns interpretability score of the feature f_name"""
        self._test_feature_exists(f_name)
        feat_dict = self.__class__.feature_descriptions[f_name]
        return feat_dict["desc"]

    def get_feature_interpretability(self, f_name):
        """Returns interpretability score of the feature f_name"""
        self._test_feature_exists(f_name)
        feat_dict = self.__class__.feature_descriptions[f_name]
        return feat_dict["interpret"]

    @classmethod
    def add_feature_description(cls, f_name, f_desc, f_interpret):
        """Adds the description to the class variable if not already there"""
        if f_name not in cls.feature_descriptions:
            cls.feature_descriptions[f_name] = {
                "desc": f_desc,
                "interpret": f_interpret,
            }

    def evaluate_feature(
        self, feature_function, feature_name, function_args=None, statistics=None,
    ):
        """Evaluating a feature function and catching/raising errors"""
        if not callable(feature_function):
            raise Exception(
                "The feature function {} is not callable!".format(feature_name)
            )
        if function_args is None:
            eval_func = partial(feature_function, self.graph)
        elif isinstance(function_args, list):
            eval_func = partial(feature_function, *function_args)
        elif isinstance(function_args, dict):
            eval_func = partial(feature_function, **function_args)
        else:
            eval_func = partial(feature_function, function_args)

        try:
            feature = eval_func()
        except (KeyboardInterrupt, SystemExit):
            sys.exit(0)
        except Exception as exc:
            L.debug(
                "Failed feature %s for graph %d with exception: %s",
                feature_name,
                self.graph.graph["id"],
                str(exc),
            )
            if statistics in ("centrality", "clustering"):
                return [np.nan]
            else:
                return np.nan
            # feature = return_type(np.nan)

        if statistics is "clustering":
            if not isinstance(feature, list):
                raise Exception(
                    "Feature {} with clustering statistics is not a list of sets: {}".format(
                        feature_name, feature
                    )
                )
            elif not isinstance(feature[0], set):
                raise Exception(
                    "Feature {} with clustering statistics is not a list of sets: {}".format(
                        feature_name, feature
                    )
                )
        elif statistics is "centrality":
            expected_types = (list, np.ndarray)
            if type(feature) not in expected_types:
                raise Exception(
                    "Feature {} with centrality statistics does not return expected type{}: {}".format(
                        feature_name, expected_types, feature
                    )
                )
        else:
            expected_types = (int, float, np.int32, np.int64, np.float32, np.float64)
            if type(feature) not in expected_types or not np.nan:
                raise Exception(
                    "Feature {} with no statistics argument does not return expected type{}: {}".format(
                        feature_name, expected_types, feature
                    )
                )

        return feature

    def add_feature(
        self,
        feature_name,
        feature_function,
        feature_description,
        feature_interpret,
        function_args=None,
        statistics=None,
    ):
        """Adds a computed feature value and its description"""
        func_result = self.evaluate_feature(
            feature_function,
            feature_name,
            function_args=function_args,
            statistics=statistics,
        )
        if statistics is None:
            self.features[feature_name] = func_result

            if not hasattr(feature_interpret, "get_score"):
                feature_interpret = InterpretabilityScore(feature_interpret)
            self.__class__.add_feature_description(
                feature_name, feature_description, feature_interpret
            )

        elif statistics == "centrality":
            self.feature_statistics(
                func_result, feature_name, feature_description, feature_interpret,
            )

        elif statistics == "clustering":
            self.clustering_statistics(
                func_result, feature_name, feature_description, feature_interpret,
            )

    def add_feature_old(
        self, feature_name, feature_function, feature_description, feature_interpret
    ):
        """Adds a computed feature value and its description"""
        if not callable(feature_function):
            raise Exception(
                "The feature function {} is not callable!".format(feature_name)
            )

        try:
            feature = feature_function(self.graph)
        except (KeyboardInterrupt, SystemExit):
            sys.exit(0)
        except Exception as exc:
            L.debug(
                "Failed feature %s for graph %d with exception: %s",
                feature_name,
                self.graph.graph["id"],
                str(exc),
            )
            # if the feature cannot be computed, fill with np.nan
            feature_trivial = feature_function(self.__class__.trivial_graph)
            if isinstance(feature_trivial, list):
                feature = [np.nan]
            else:
                feature = np.nan

        if isinstance(feature, (list, np.ndarray)):
            if isinstance(feature[0], set):
                self.clustering_statistics(
                    feature, feature_name, feature_description, feature_interpret
                )
            else:
                self.feature_statistics(
                    feature, feature_name, feature_description, feature_interpret
                )
        else:
            self.features[feature_name] = feature
            if not hasattr(feature_interpret, "get_score"):
                feature_interpret = InterpretabilityScore(feature_interpret)
            self.__class__.add_feature_description(
                feature_name, feature_description, feature_interpret
            )

    def compute_features(self):
        """main feature extraction function"""
        self.add_feature(
            "test", lambda graph: 0.0, "Test feature for the base feature class", 5
        )

    def update_features(self, all_features):
        """update the feature dictionary if correct mode provided"""

        if (
            self.__class__.shortname == "TP"
            and self.__class__.__name__ != "FeatureClass"
        ):
            raise Exception(
                "Shortname not set for feature class {}".format(self.__class__.__name__)
            )

        self.compute_features()

        if self.normalize_features:
            self.compute_normalize_features()

        all_features[self.__class__.shortname] = self.features

    def compute_normalize_features(self):
        """triple the number of features by normalising by node and edges"""

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

    def clustering_statistics(
        self, community_partition, feat_name, feat_desc, feat_interpret
    ):
        """ Compute quality of the community partitions """
        compl_desc = " of the partition of " + feat_desc

        self.add_feature(
            feat_name + "_modularity",
            quality.modularity,
            "Modularity" + compl_desc,
            feat_interpret,
            function_args=community_partition,
        )
        self.add_feature(
            feat_name + "_coverage",
            quality.coverage,
            "Coverage" + compl_desc,
            feat_interpret,
            function_args=community_partition,
        )
        self.add_feature(
            feat_name + "_performance",
            quality.performance,
            "Performance" + compl_desc,
            feat_interpret,
            function_args=community_partition,
        )
        self.add_feature(
            feat_name + "_inter_community_edges",
            quality.inter_community_edges,
            "Inter community edges" + compl_desc,
            feat_interpret,
            function_args=community_partition,
        )
        self.add_feature(
            feat_name + "_inter_community_non_edges",
            quality.inter_community_non_edges,
            "Inter community non edges" + compl_desc,
            feat_interpret,
            function_args=community_partition,
        )
        self.add_feature(
            feat_name + "_intra_community_edges",
            quality.intra_community_edges,
            "Intra community edges" + compl_desc,
            feat_interpret,
            function_args=community_partition,
        )

    def feature_statistics(self, feat_dist, feat_name, feat_desc, feat_interpret):
        """Computes summary statistics of distributions"""

        self.feature_statistics_basic(feat_dist, feat_name, feat_desc, feat_interpret)

        if self.statistics_level == "medium":
            self.feature_statistics_medium(
                feat_dist, feat_name, feat_desc, feat_interpret
            )

        if self.statistics_level == "advanced":
            self.feature_statistics_medium(
                feat_dist, feat_name, feat_desc, feat_interpret
            )
            self.feature_statistics_advanced(
                feat_dist, feat_name, feat_desc, feat_interpret
            )

    def feature_statistics_basic(self, feat_dist, feat_name, feat_desc, feat_interpret):
        """Computes basic summary statistics of distributions"""
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

    def feature_statistics_medium(
        self, feat_dist, feat_name, feat_desc, feat_interpret
    ):
        """Computes mediumsummary statistics of distributions"""
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

    def feature_statistics_advanced(
        self, feat_dist, feat_name, feat_desc, feat_interpret
    ):
        """Computes advanced summary statistics of distributions"""
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
        self.add_feature(
            feat_name + "_bayes_confint",
            lambda dist: st.bayes_mvs(dist)[0][1][1] - st.bayes_mvs(dist)[0][1][0],
            "Bayes confidance interval" + compl_desc,
            feat_interpret - 1,
            function_args=feat_dist,
        )


class InterpretabilityScore:
    min_score = 0
    max_score = 5

    def __init__(self, score):
        """
        Init function for InterpretabilityScore

        Parameters
        ----------
        score: number or {'min', 'max'}
            value of score to set
            will be shrunk to be within [min_score, max_score]
        """
        if score == "max":
            score = self.max_score
        elif score == "min":
            score = self.min_score
        self.set_score(score)

    def set_score(self, score):
        score_int = int(score)
        if score_int < self.min_score:
            score_int = self.min_score
        elif score_int > self.max_score:
            score_int = self.max_score
        self.score = score_int

    def get_score(self):
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
