from __future__ import print_function, unicode_literals
import tensorflow as tf
import os

from pose_network.core.Types import *
from pose_network.nets.RatNetMultiView import RatNetMultiView


def load_pose_network(cfg, model, return_cfg=False):
    # derive some parameters from the config files
    sizes = [len(d['cam_range']) for d in model.datasets]
    assert all([sizes[0] == s for s in sizes]), 'Datasets have different number of cameras. This should not be the case.'
    cfg['batch_size'] = sizes[0]
    cfg['num_kp'] = len(model.keypoints)

    # placeholders
    image = tf.placeholder(dtype=tf.float32, shape=(cfg['batch_size'], 224, 224, 3), name='image')
    K = tf.placeholder(dtype=tf.float32, shape=(cfg['batch_size'], 3, 3), name='K')
    M = tf.placeholder(dtype=tf.float32, shape=(cfg['batch_size'], 4, 4), name='M')
    root = tf.placeholder(dtype=tf.float32, shape=(1, 1, 3), name='root')

    # turn dictionary with keys into a class with fields
    class classify(object):
        def __init__(self, cfg):
            for k, v in cfg.items():
                setattr(self, k, v)

    # network
    net = RatNetMultiView(classify(cfg))
    pred, _ = net.inference({data_t.image: image,
                             data_t.K: K,
                             data_t.M: M,
                             data_t.voxel_root: root}, {}, is_training=False)

    # setup session
    config = tf.ConfigProto(allow_soft_placement=True)
    session = tf.Session(config=config)
    saver = tf.train.Saver()
    last_file = tf.train.latest_checkpoint(model.pose_models[-1])
    assert last_file is not None, "No snapshot found!"
    saver.restore(session, last_file)
    print('Snapshot restored from: %s' % last_file)

    inputs = {
        'image': image,
        'K': K,
        'M': M,
        'root': root
    }

    if return_cfg:
        return session, inputs, pred, cfg
    else:
        return session, inputs, pred
