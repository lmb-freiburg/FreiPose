from __future__ import print_function, unicode_literals
import tensorflow as tf
import numpy as np

from .Types import *
from utils.general_util import listify
from utils.Scoremaps import create_scorevol_3d_tf_batch, create_scoremaps_2d_tf_batch


def _get(xyz, ids):
    if type(ids) == list:
        c = tf.add_n([xyz[:, i] for i in ids]) /float(len(ids))
        return c
    else:
        return xyz[:, ids]


def loss_factory(config, loss_item, ground_truths, predictions):
    """ Calculates some kind of loss between ground truth and predictions. """
    loss = 0.0
    gt = ground_truths[loss_item.gt] if loss_item.gt in ground_truths.keys() else None

    # if gt is None and 'pred' in loss_item.gt:
    #     gt = predictions[loss_item.gt]

    if type(loss_item.pred) == list:
        # there is more than one prediction item involved
        pred = [listify(predictions[p]) for p in loss_item.pred]
        pred = zip(*pred)  # change order of inner and outer list

    else:
        # more usual case: there is one (list of) prediction
        pred = listify(predictions[loss_item.pred])

    # weight this loss component
    loss_weight = loss_item.weight
    if not type(loss_weight) == list:
        loss_weight = [loss_weight for _ in pred]
    assert len(loss_weight) == len(pred), 'Number of loss weights and predictions must be equal.'

    # create scoremap for scoremap loss
    if loss_item.type == loss_t.l2_scoremap:
        gt = create_scoremaps_2d_tf_batch(gt,
                                          sigma=10.0,
                                          img_shape_hw=[config.img_size]*2,
                                          valid_vec=ground_truths[data_t.vis_merged])

    # create scorevolume for scorevolume loss
    if loss_item.type == loss_t.l2_scorevolume:
        gt = create_scorevol_3d_tf_batch(gt,
                                         sigma=config.scorevol_sigma,
                                         output_size=[config.voxel_dim]*3,
                                         valid_vec=ground_truths[data_t.vis_nobatch])

    for w, p in zip(loss_weight, pred):

        if loss_item.type == loss_t.l2:
            # # Simple elementwise L2 loss btw gt and prediction
            # this_loss = tf.sqrt(0.5 * tf.losses.mean_squared_error(gt,
            #                                                    p)
            #                 + 1e-8)

            # squared L2 loss
            this_loss = tf.sqrt(tf.square(gt-p) + 1e-8)
            this_loss = tf.reduce_mean(this_loss)

            loss += w * this_loss  # weight and accumulate

        elif loss_item.type == loss_t.l2_scorevolume:
            vis = tf.reshape(ground_truths[data_t.vis_nobatch], [1, 1, 1, 1, -1])
            this_loss = 0.5 * tf.losses.mean_squared_error(gt, p, vis)
            # this_loss = 0.5 * tf.losses.mean_squared_error(gt, p)
            # this_loss = tf.reduce_mean(ground_truths[data_t.vis_nobatch] * tf.reduce_mean(tf.square(gt - p), [1, 2, 3]))
            # this_loss = tf.reduce_mean(tf.reduce_mean(tf.square(gt - p), [1, 2, 3]))

            loss += w * this_loss  # weight and accumulate

        elif loss_item.type == loss_t.l2_scoremap:
            # vis = tf.expand_dims(tf.expand_dims(ground_truths[data_t.vis_merged], 1), 1)
            # this_loss = 0.5 * tf.losses.mean_squared_error(gt, p, vis)

            # this_loss = 0.5 * tf.losses.mean_squared_error(gt, p)
            # this_loss = tf.reduce_mean(ground_truths[data_t.vis_merged] * tf.reduce_mean(tf.square(gt - p), [1, 2]))
            # this_loss = tf.reduce_mean(tf.reduce_mean(tf.square(gt - p), [1, 2]))

            # squared L2 loss
            this_loss = tf.sqrt(tf.square(gt-p) + 1e-8)

            # mask by visibilty of keypoints
            vis = ground_truths[data_t.vis_merged]
            this_loss = vis * tf.reduce_mean(this_loss, [1, 2])
            this_loss = tf.reduce_mean(this_loss)

            
            loss += w * this_loss  # weight and accumulate

        elif loss_item.type == loss_t.limb_length:
            this_loss = 0.0
            for i, (pid, cid) in enumerate(config.limbs):
                limb_pred = tf.norm(_get(p, pid) - _get(p, cid), ord=2, axis=-1)
                this_lmb_loss = tf.maximum(config.limb_limits[i][0] - limb_pred, 0.0)  # check lower limit
                this_lmb_loss += tf.maximum(limb_pred - config.limb_limits[i][1], 0.0)  # check upper limit
                this_loss += tf.reduce_mean(this_lmb_loss)

            loss += w * this_loss  # weight and accumulate

        elif loss_item.type == loss_t.limb_angles:
            this_loss = 0.0
            for i, (lid1, lid2) in enumerate(config.kinematic_dep):
                # get limbs
                cid, pid = config.limbs[lid1]
                limb1 = _get(p, cid) - _get(p, pid)
                cid, pid = config.limbs[lid2]
                limb2 = _get(p, cid) - _get(p, pid)

                # normalize length
                limb1 /= tf.norm(limb1, ord=2, axis=-1)
                limb2 /= tf.norm(limb2, ord=2, axis=-1)

                # calculate cos(angle)
                angle_pred = tf.reduce_sum(tf.multiply(limb1, limb2), -1)

                this_lmb_loss = tf.maximum(config.limb_angle_limits[i][0] - angle_pred, 0.0)  # check lower limit
                this_lmb_loss += tf.maximum(angle_pred - config.limb_angle_limits[i][1], 0.0)  # check upper limit
                this_loss += tf.reduce_mean(this_lmb_loss)

            loss += w * this_loss  # weight and accumulate

        else:
            print('Unknown loss item:', loss_item)
            raise NotImplementedError

    return loss

