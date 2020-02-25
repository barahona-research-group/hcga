"""template class for feature extraction"""
from ..feature_utils import normalize_features

class FeatureClass():
    """template class"""

    # Class variables that describe the feature, 
    # They should be defined for all child features
    normalize_features = True
    modes = ['fast', 'medium', 'slow']
    shortname = 'TP'
    name = 'template'
    keywords = ['template']

    # Feature descriptions as class variable
    feature_descriptions = {}

    def __init_subclass__(cls, **kwargs):
        """Initialise class variables to default for each child class"""
        #super().__init_subclass__(**kwargs)
        cls.feature_descriptions = {}

    def __init__(self, graph=None):
        """init function"""
        self.graph = graph
        self.features = {}

    def get_info(self):
        """return a dictionary of informations about the feature class"""
        return {'name': self.__class__.name, 
                'shortname': self.__class__.shortname,
                'keywords': self.__class__.keywords
                }

    def _test_feature_exists(self, f_name):
        """Test if feature f_name exists in description list"""
        if f_name not in  self.__class__.feature_descriptions:
            raise Exception('Feature {} does not exist in class {}'.format(
                f_name, self.name))

    def get_feature_info(self, f_name):
        """Returns a dictionary of information about the feature f_name"""
        self._test_feature_exists(f_name)
        feat_info = self.get_info()
        feat_info['feature_name'] = f_name
        feat_dict = self.__class__.feature_descriptions[f_name]
        feat_info['feature_description'] = feat_dict['desc']
        feat_info['feature_interpretability'] = feat_dict['interpret']
        return feat_info

    def get_feature_description(self, f_name):
        """Returns interpretability score of the feature f_name"""
        self._test_feature_exists(f_name)
        feat_dict = self.__class__.feature_descriptions[f_name]
        return feat_dict['desc']

    def get_feature_interpretability(self, f_name):
        """Returns interpretability score of the feature f_name"""
        self._test_feature_exists(f_name)
        feat_dict = self.__class__.feature_descriptions[f_name]
        return feat_dict['interpret']

    @classmethod
    def add_feature_description(cls, f_name, f_desc, f_interpret):
        """Adds the description to the class variable if not already there"""
        if f_name not in cls.feature_descriptions:
            cls.feature_descriptions[f_name] = {
                    'desc':f_desc, 'interpret':f_interpret}

    def add_feature(self, feature_name, feature_value, 
            feature_description, feature_interpret):
        """Adds a computed feature value and its description"""
        if not hasattr(feature_interpret, 'get_score'):
            feature_interpret = InterpretabilityScore(feature_interpret)
        self.features[feature_name] = feature_value
        self.__class__.add_feature_description(
                feature_name, feature_description, feature_interpret)

    def compute_features(self):
        """main feature extraction function"""
        self.add_feature('test', 0., 
                'Test feature for the base feature class', 5
                )

    def update_features(self, all_features):
        """update the feature dictionary if correct mode provided"""
        if self.graph is None:
            raise Exception('No graph provided to compute some features.')
        if self.__class__.shortname == 'TP' and self.__class__.__name__ != 'FeatureClass':
            raise Exception('Shortname not set for feature class {}'.format(self.__class__.__name__))

        self.compute_features()
        if self.__class__.normalize_features:
            normalize_features(self)
        all_features[self.__class__.shortname] = self.features

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
        if score == 'max':
            score = self.max_score
        elif score == 'min':
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
        else: othervalue = other.get_score()
        result = self.__class__(self.get_score() + othervalue)
        return result

    def __sub__(self, other):
        if type(other) == int:
            othervalue = other
        else: othervalue = other.get_score()
        result = self.__class__(self.get_score() - othervalue)
        return result

    def __str__(self):
        return str(self.score)

    def __repr__(self):
        return self.__class__.__name__ + '({})'.format(self.__str__())

