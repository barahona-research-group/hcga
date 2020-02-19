"""template class for feature extraction"""
from ..feature_utils import normalize_features

class FeatureClass():
    """template class"""

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

    def compute_features(self):
        """main feature extraction function"""
        self.features['test'] = 0.

    def update_features(self, all_features, mode):
        """update the feature dictionary if correct mode provided"""
        if self.graph is None:
            raise Exception('No graph provided to compute some features.')

        if mode in self.modes:
            self.compute_features()
            if self.normalize_features:
                normalize_features(self.features, self.graph)
            all_features[self.shortname] = self.features

