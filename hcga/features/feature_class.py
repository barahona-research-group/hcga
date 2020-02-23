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

    def get_feature_info(self, f_name):
        """Returns a dictionary of information about the feature f_name"""
        if f_name not in  self.__class__.feature_descriptions:
            raise Exception('Feature {} does not exist in class {}'.format(
                f_name, self.name))
        feat_dict = self.get_info()
        feat_dict['feature_name'] = f_name
        feat_dict['feature_description'] = self.__class__.feature_descriptions[f_name]
        return feat_dict

    @classmethod
    def add_feature_description(cls, f_name, f_desc):
        """Adds the description to the class variable if not already there"""
        if f_name not in cls.feature_descriptions:
            cls.feature_descriptions[f_name] = f_desc

    def add_feature(self, feature_name, feature_value, feature_description):
        """Adds a computed feature value and its description"""
        self.features[feature_name] = feature_value
        self.__class__.add_feature_description(feature_name, feature_description)

    def compute_features(self):
        """main feature extraction function"""
        self.add_feature('test', 0., 'Test feature for the base feature class')

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

