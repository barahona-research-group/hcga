"""template class for feature extraction"""
from ..feature_utils import normalize_features

class FeatureClass():
    """template class"""

    # Feature descriptions as class variable
    feature_descriptions = {}

    def __init_subclass__(cls, **kwargs):
        """Initialise list of feature descriptions to zero for each child class"""
        super().__init_subclass__(**kwargs)
        cls.feature_descriptions = {}

    def __init__(self, graph=None):
        """init function"""
        self.graph = graph
        self.features = {}
        self.set_infos()
        self.normalize_features = True

    def set_infos(self):
        """set class infos"""
        self.modes = ['fast', 'medium', 'slow']
        self.shortname = 'TP'
        self.name = 'template'
        self.keywords = 'template'

    def get_info(self):
        """return a dictionary of feature informations"""
        return {'name': self.name, 
                'shortname': self.shortname,
                'keywords': self.keywords
                }

    def get_feature_info(self, f_name):
        if f_name not in  self.__class__.feature_descriptions:
            raise Exception('Feature {} does not exist in class {}'.format(
                f_name, self.name))
        feat_dict = self.get_info()
        feat_dict['feature_name'] = f_name
        feat_dict['feature_description'] = self.__class__.feature_descriptions[f_name]
        return feat_dict


    @classmethod
    def add_feature_description(cls, f_name, f_desc):
        if f_name not in cls.feature_descriptions:
            cls.feature_descriptions[f_name] = f_desc

    def add_feature(self, feature_name, feature_value, feature_description):
        """Adds a computed feature value and its description"""
        self.features[feature_name] = feature_value
        # Adds the description in the class variable if not already there
        self.__class__.add_feature_description(feature_name, feature_description)

    def compute_features(self):
        """main feature extraction function"""
        self.add_feature('test', 0., 'Test feature for the base feature class')

    def update_features(self, all_features, mode):
        """update the feature dictionary if correct mode provided"""
        if self.graph is None:
            raise Exception('No graph provided to compute some features.')

        if mode in self.modes:
            self.compute_features()
            if self.normalize_features:
                normalize_features(self)
            all_features[self.shortname] = self.features

