from __future__ import print_function, unicode_literals
import tensorflow as tf

from .Types import *


def ph_factory(config, input_list):
    """ Returns a dict of placeholders that are requested by input_list, which is a  list of data_t items. """
    
    inputs = dict()
    for item in input_list:
        if item == data_t.image:
            inputs[data_t.image] = tf.placeholder(shape=[config.batch_size,
                                                         config.img_size,
                                                         config.img_size, 3],
                                                  dtype=tf.float32, name='image')
        elif item == data_t.K:
            inputs[data_t.K] = tf.placeholder(shape=[config.batch_size, 3, 3],
                                              dtype=tf.float32, name='cam_intrinsic')
        elif item == data_t.M:
            inputs[data_t.M] = tf.placeholder(shape=[config.batch_size, 4, 4],
                                              dtype=tf.float32, name='cam_extrinsic')

        elif item == data_t.xyz_nobatch or item == data_t.xyz_vox_nobatch:
            inputs[item] = tf.placeholder(shape=[1, config.num_kp, 3],
                                                dtype=tf.float32, name=item)
        elif item == data_t.uv or item == data_t.uv_merged:
            inputs[item] = tf.placeholder(shape=[config.batch_size, config.num_kp, 2],
                                               dtype=tf.float32, name=item)
        elif item == data_t.vis_merged:
            inputs[item] = tf.placeholder(shape=[config.batch_size, config.num_kp],
                                                dtype=tf.float32, name=item)
        elif item == data_t.vis_nobatch:
            inputs[item] = tf.placeholder(shape=[1, config.num_kp],
                                                dtype=tf.float32, name=item)
        elif item == data_t.voxel_root:
            inputs[data_t.voxel_root] = tf.placeholder(shape=[1, 1, 3],
                                                       dtype=tf.float32, name='voxel_root')
        elif item == data_t.is_supervised:
            inputs[data_t.is_supervised] = tf.placeholder(shape=[1],
                                                       dtype=tf.float32, name='is_supervised')
        else:
            print('Unknown placeholder item: ', item)
            raise NotImplementedError

    return inputs
