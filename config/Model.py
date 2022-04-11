from __future__ import unicode_literals, print_function
from utils.general_util import json_load


class Model(object):
    """ A FreiPose Model object encapsulates a predictive model and stores associated information. This includes:
            - Skeletal definition of landmarks predicted
            - Labeled data
            - Trained models

        It can be loaded from two configuration files.
    """
    def __init__(self, model_file):
        """ Possibly load model from file. """
        self.model_file = model_file
        model_data = json_load(model_file)
        self.keypoints, self.limbs, self.viewer, self.coord_frames, self.body_angles = self.load_def(model_data['skeleton'])
        self.datasets, self.bb_models, self.pose_models, self.preprocessing = self.load_data(model_data['data'])

    def load_def(self, def_file):
        """ Load model from file. """
        data = json_load(def_file)
        self.data = data
        return data['keypoints'], data['limbs'], data['viewer'], data['coord_frames'], data['body_angles']

    def load_data(self, data_file):
        """ Load model from file. """
        data = json_load(data_file)

        assert 'bb_networks' in data.keys(), 'Missing needed field.'
        assert 'pose_networks' in data.keys(), 'Missing needed field.'
        assert 'datasets' in data.keys(), 'Missing needed field.'
        db_field = ['anno', 'calib', 'cam_range', 'db_set', 'frame_dir', 'path', 'vid_file']
        assert all([all([k in d.keys() for k in db_field]) for d in data['datasets']]), 'Missing needed field.'

        return data['datasets'], data['bb_networks'], data['pose_networks'], data['preprocessing']

    def __repr__(self):
        return self.print_me()

    def __str__(self):
        return self.print_me()

    def print_me(self):
        return '<FreiPose.Model with model_file: %s>' % self.model_file

