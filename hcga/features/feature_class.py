"""template class for feature extraction"""
import numpy as np
import scipy.stats as st
import networkx as nx
from networkx.algorithms.community import quality


class FeatureClass:
    """template class"""

    # Class variables that describe the feature,
    # They should be defined for all child features
    modes = ["fast", "medium", "slow"]
    shortname = "TP"
    name = "template"
    keywords = ["template"]

    statistics_level = 'basic'
    normalize_features = True

    # Feature descriptions as class variable
    feature_descriptions = {}

    trivial_graph = nx.generators.classic.complete_graph(3)

    def __init_subclass__(cls, **kwargs):
        """Initialise class variables to default for each child class"""
        # super().__init_subclass__(**kwargs)
        cls.feature_descriptions = {}

    def __init__(self, graph=None):
        """init function"""
        self.graph = graph
        self.features = {}

    @classmethod
    def setup_class(cls, normalize_features=True, statistics_level='basic'):
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

    def add_feature(
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
            raise
        except:
            # if the feature cannot be computed, fill with np.nan
            feature_trivial = feature_function(self.__class__.trivial_graph)
            if isinstance(feature_trivial, list):
                feature = [np.nan]
            else:
                feature = np.nan

        if isinstance(feature, list) or isinstance(feature, np.ndarray):
            # if the feature is a list of numbers, extract statistics
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
        self.add_feature("test", lambda graph: 0.0, 
                "Test feature for the base feature class", 5)

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

    def clustering_statistics(self, community_partition, feat_name, feat_desc, feat_interpret):
        """ Compute quality of the community partitions """
        
        compl_desc = " of the partition of " + feat_desc

        
        self.add_feature(
            feat_name + "_modularity",
            lambda graph: _try(quality.modularity, community_partition),
            "Modularity" + compl_desc,
            feat_interpret,
        )
        self.add_feature(
            feat_name + "_coverage",
            lambda graph: _try(quality.coverage, community_partition),
            "Coverage" + compl_desc,
            feat_interpret,
        )
        self.add_feature(
            feat_name + "_performance",
            lambda graph: _try(quality.performance, community_partition),
            "Performance" + compl_desc,
            feat_interpret,
        )
        self.add_feature(
            feat_name + "_inter_community_edges",
            lambda graph: _try(quality.inter_community_edges, community_partition),
            "Inter community edges" + compl_desc,
            feat_interpret,
        )        
        self.add_feature(
            feat_name + "_inter_community_non_edges",
            lambda graph: _try(quality.inter_community_non_edges, community_partition),
            "Inter community non edges" + compl_desc,
            feat_interpret,
        )        
        self.add_feature(
            feat_name + "_intra_community_edges",
            lambda graph: _try(quality.intra_community_edges, community_partition),
            "Intra community edges" + compl_desc,
            feat_interpret,
        )                        


    def feature_statistics(self, feat_dist, feat_name, feat_desc, feat_interpret):
        """Computes summary statistics of distributions"""

        self.feature_statistics_basic(feat_dist, feat_name, feat_desc, feat_interpret)

        if self.statistics_level == 'medium':
            self.feature_statistics_medium(feat_dist, feat_name, feat_desc, feat_interpret)

        if self.statistics_level == 'advanced':
            self.feature_statistics_medium(feat_dist, feat_name, feat_desc, feat_interpret)
            self.feature_statistics_advanced(feat_dist, feat_name, feat_desc, feat_interpret)

    def feature_statistics_basic(self, feat_dist, feat_name, feat_desc, feat_interpret):
        """Computes basic summary statistics of distributions"""
        compl_desc = " of the distribution of " + feat_desc

        self.add_feature(
            feat_name + "_mean",
            lambda graph: _try(np.mean, feat_dist),
            "Mean" + compl_desc,
            feat_interpret,
        )
        self.add_feature(
            feat_name + "_min",
            lambda graph: _try(np.min, feat_dist),
            "Minimum" + compl_desc,
            feat_interpret,
        )
        self.add_feature(
            feat_name + "_max",
            lambda graph: _try(np.max, feat_dist),
            "Maximum" + compl_desc,
            feat_interpret,
        )
        self.add_feature(
            feat_name + "_median",
            lambda graph: _try(np.median, feat_dist),
            "Median" + compl_desc,
            feat_interpret,
        )
        self.add_feature(
            feat_name + "_std",
            lambda graph: _try(np.std, feat_dist),
            "Standard deviation" + compl_desc,
            feat_interpret,
        )

    def feature_statistics_medium(self, feat_dist, feat_name, feat_desc, feat_interpret):
        """Computes mediumsummary statistics of distributions"""
        compl_desc = " of the distribution of " + feat_desc

        self.add_feature(
            feat_name + "_gmean",
            lambda graph: _try(st.gmean, feat_dist),
            "G. Mean" + compl_desc,
            feat_interpret - 1,
        )
        self.add_feature(
            feat_name + "_hmean",
            lambda graph: _try(lambda dist: st.hmean(np.abs(dist) + 1e-8), feat_dist),
            "G. Mean" + compl_desc,
            feat_interpret - 1,
        )
        self.add_feature(
            feat_name + "_kurtosis",
            lambda graph: _try(st.kurtosis, feat_dist),
            "Kurtosis" + compl_desc,
            feat_interpret - 1,
        )
        self.add_feature(
            feat_name + "_mode",
            lambda graph: _try(lambda dist: st.mode(dist)[0][0], feat_dist),
            "Mode" + compl_desc,
            feat_interpret - 1,
        )

    def feature_statistics_advanced(self, feat_dist, feat_name, feat_desc, feat_interpret):
        """Computes advanced summary statistics of distributions"""
        compl_desc = " of the distribution of " + feat_desc

        self.add_feature(
            feat_name + "_kstat",
            lambda graph: _try(st.kstat, feat_dist),
            "Kstat" + compl_desc,
            feat_interpret - 1,
        )
        self.add_feature(
            feat_name + "_kstatvar",
            lambda graph: _try(st.kstatvar, feat_dist),
            "Kstat variance" + compl_desc,
            feat_interpret - 1,
        )
        self.add_feature(
            feat_name + "_tmean",
            lambda graph: _try(st.tmean, feat_dist),
            "Tmean" + compl_desc,
            feat_interpret - 1,
        )
        self.add_feature(
            feat_name + "_tmean",
            lambda graph: _try(st.tvar, feat_dist),
            "Tvariance" + compl_desc,
            feat_interpret - 1,
        )
        self.add_feature(
            feat_name + "_tmin",
            lambda graph: _try(st.tmin, feat_dist),
            "Tminimum" + compl_desc,
            feat_interpret - 1,
        )
        self.add_feature(
            feat_name + "_tmax",
            lambda graph: _try(st.tmax, feat_dist),
            "Tmaximum" + compl_desc,
            feat_interpret - 1,
        )
        self.add_feature(
            feat_name + "_tstd",
            lambda graph: _try(st.tstd, feat_dist),
            "T standard deviation" + compl_desc,
            feat_interpret - 1,
        )
        self.add_feature(
            feat_name + "_tsem",
            lambda graph: _try(st.tsem, feat_dist),
            "T sem" + compl_desc,
            feat_interpret - 1,
        )
        self.add_feature(
            feat_name + "_variation",
            lambda graph: _try(st.variation, feat_dist),
            "Variation" + compl_desc,
            feat_interpret - 1,
        )
        self.add_feature(
            feat_name + "_mean_repeats",
            lambda graph: _try(
                lambda dist: np.mean(st.find_repeats(dist)[1]), feat_dist
            ),
            "Mean repeats" + compl_desc,
            feat_interpret - 1,
        )
        self.add_feature(
            feat_name + "_entropy",
            lambda graph: _try(st.entropy, feat_dist),
            "Entropy" + compl_desc,
            feat_interpret - 1,
        )
        self.add_feature(
            feat_name + "_sem",
            lambda graph: _try(st.sem, feat_dist),
            "SEM" + compl_desc,
            feat_interpret - 1,
        )
        self.add_feature(
            feat_name + "_bayes_confint",
            lambda graph: _try(
                lambda dist: st.bayes_mvs(dist)[0][1][1] - st.bayes_mvs(dist)[0][1][0],
                feat_dist,
            ),
            "Bayes confidance interval" + compl_desc,
            feat_interpret - 1,
        )

def _try(func, feat_dist):
    try:
        return func(feat_dist)
    except (KeyboardInterrupt, SystemExit):
        raise
    except:
        return np.nan


class InterpretabilityScore:
    min_score = 0
    max_score = 5

    def __init__(self, score):
        """
        Init function for InterpretabilityScore

        Parameters
        ----------
        score: number or {'min', 'max'}
            value of score to set, 
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
        if type(other) == int:
            othervalue = other
        else:
            othervalue = other.get_score()
        result = self.__class__(self.get_score() + othervalue)
        return result

    def __sub__(self, other):
        if type(other) == int:
            othervalue = other
        else:
            othervalue = other.get_score()
        result = self.__class__(self.get_score() - othervalue)
        return result

    def __str__(self):
        return str(self.score)

    def __repr__(self):
        return self.__class__.__name__ + "({})".format(self.__str__())
