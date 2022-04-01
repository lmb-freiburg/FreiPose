from __future__ import unicode_literals, print_function
from utils.general_util import json_load


class Param(object):
    """ A FreiPose Param object encapsulates parameters steering network training.
    """
    def __init__(self, bb_file=None, pose_file=None, viewer_file=None):
        """ Possibly load model from file. """
        if bb_file is None:
            bb_file = 'config/bb_network.cfg.json'
        if pose_file is None:
            pose_file = 'config/pose_network.cfg.json'
        if viewer_file is None:
            viewer_file = 'config/viewer.cfg.json'

        self.bb = json_load(bb_file)
        self.pose = json_load(pose_file)
        self.viewer = json_load(viewer_file)
