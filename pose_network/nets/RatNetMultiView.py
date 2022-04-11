from __future__ import print_function, unicode_literals
import tensorflow as tf
import tensorflow.contrib.slim as slim
from tqdm import tqdm

from tensorflow.contrib.layers.python.layers.initializers import variance_scaling_initializer

"""
    Network for keypoint predictions based on color information.
"""

from pose_network.core.Types import *
from pose_network.nets.NetworkBase import *
from utils.triang import *

from pose_network.perspective.unproject import unproject as perspective_unproject
from utils.Scoremaps import argmax_3d, create_scoremaps_2d_tf_batch, argmax_2d
from utils.Softargmax import softargmax3D, softargmax2D

from .NetworkOps import NetworkOps as ops


t_tool = None
def _triangulate(points, confidences, K_list, M_list):
    global t_tool
    if t_tool is None:
        t_tool = TriangTool()
    points3d, _, vis3d, _, _ = triangulate_robust(points, confidences > 0.25, K_list, M_list, threshold=50.0)

    return np.expand_dims(points3d, 0), np.expand_dims(vis3d, 0)


class RatNetMultiView(NetworkBase):
    def __init__(self, net_config):
        super(RatNetMultiView, self).__init__(net_config)

        # these to lists are used to create placeholders
        self.needed_inputs = [data_t.image,  # 8 cameras representing one time instance
                              data_t.K,  # camera calibration for unprojecting
                              data_t.M,
                              data_t.voxel_root  # where in the world our voxel is centered
                              ]
        self.needed_ground_truths = [data_t.xyz_nobatch,  # for 3D keypoint loss
                                     data_t.vis_nobatch,
                                     data_t.uv_merged,  # for 2D keypoint loss
                                     data_t.vis_merged,
                                     data_t.xyz_vox_nobatch
                                     ]

        if net_config.semisup:
            self.needed_ground_truths.append(data_t.is_supervised)

        self.config = net_config

        # flag step (misusing learningrate decay scheduler, which is setup during creating the optimizer)
        self.flag_ref = None

    def inference(self, input, predictions, is_training=True):
        """ Inference part of the network at training time. """
        # run inference
        scorevol_pred_list, xyz_pred_list, score3d_list,\
        scoremap_pred_refined, pred_uv_refine,\
        pred_3d_refine, pred_3d_vis_refine, \
        variables = self.inference_core(input[data_t.image],
                                        input[data_t.K], input[data_t.M], input[data_t.voxel_root],
                                        is_training=is_training)

        # project points into the image: use GT root keypoint
        batch = input[data_t.image].get_shape().as_list()[0]
        xyz_pred_final = tf.tile(xyz_pred_list[-1], [batch, 1, 1])
        xyz_pred_cam = trafoPoints(xyz_pred_final, input[data_t.M])
        uv_pred = projectPoints(xyz_pred_cam, input[data_t.K])

        self._all_vars.extend(variables)

        return {data_t.pred_score3d: score3d_list,  # some measure of certainty / visibility
                data_t.pred_uv_final: uv_pred,  # projected prediction from the 3D network
                data_t.pred_xyz: xyz_pred_list, # prediction of the 3D network
                data_t.pred_xyz_final: xyz_pred_list[-1],
                data_t.pred_scorevol: scorevol_pred_list,
                data_t.pred_uv_refine: pred_uv_refine,
                data_t.pred_scoremap: scoremap_pred_refined,
                data_t.pred_xyz_refine: pred_3d_refine,
                data_t.pred_vis3d_refine: pred_3d_vis_refine
                }, variables

    def inference_test(self, input, is_training=True):
        """ Inference part of the network at test time. """
        return self.inference(input, is_training)

    def inference_core(self, image, K_tf, M_tf, voxel_root, is_training):
        """ Inference part of the network. """
        dev1 = 'GPU:0'
        dev2 = 'GPU:0'

        variables = list()
        with tf.device(dev1):
            # Run 2D encoder
            encoding2d, enc_skips, var_2d = self._inference2D(image, trainable=self.config.train_2d_encoder)
            self._pretrained_vars = var_2d
            variables.extend(var_2d)

            # compensate for missing upsampling of encoding
            K_ds_tf = tf.stack([K_tf[:, 0, :]/8.0, K_tf[:, 1, :]/8.0, K_tf[:, 2, :]], 1)

            # if self.config.compress_2d:
            #     with tf.name_scope('compression'):
            #         encoding = slim.conv2d(encoding, num_outputs=64, kernel_size=1, stride=1, activation_fn=None)

            # Unproject encoding into common voxel
            root_tiles = tf.tile(voxel_root, [self.net_config.batch_size, 1, 1])
            volume, coords_xyz = perspective_unproject(encoding2d,
                                                       K_ds_tf, M_tf,
                                                       root_tiles,
                                                       self.config.voxel_dim, self.config.voxel_resolution)

            with tf.name_scope('mean_unprojected_feats'):
                volume = tf.reduce_mean(volume, 0)  # reduce features over all views
                volume = tf.expand_dims(volume, 0)

            if not self.config.train_2d_encoder:
                volume = tf.stop_gradient(volume)

        # 3D CNN hourglass to localize keypoints
        with tf.device(dev2):
            scorevol_pred_list, xyz_vox_pred_list, score_list, var_3d = self._inference3D(volume, is_training)
            variables.extend(var_3d)

            # transform into metric coordinates
            xyz_pred_list = [self._vox2metric(xyz_vox, voxel_root) for xyz_vox in xyz_vox_pred_list]

        scoremap_pred_refined, pred_uv_refine = list(), list()
        pred_3d_refine, pred_3d_vis_refine = None, None
        if self.net_config.use_2drefinement_net:
            # 2D CNN hourglass to refine keypoints
            with tf.device(dev2):
                s = encoding2d.get_shape().as_list()

                # project final 3D hypothesis into the views
                xyz_pred_final = tf.tile(xyz_pred_list[-1], [s[0], 1, 1])
                xyz_pred_cam = trafoPoints(xyz_pred_final, M_tf)
                uv_pred = projectPoints(xyz_pred_cam, K_ds_tf)  # project into the downsampled image frame

                # Create 2D scoremaps based on the projected prediction the 3D net did
                scoremap_pred = create_scoremaps_2d_tf_batch(uv_pred, sigma=2.0, img_shape_hw=(s[1], s[2]))

                # refine 2D predictions based on image features
                scoremap_pred_refined, var_ref = self._inference2D_refine(scoremap_pred, enc_skips)
                variables.extend(var_ref)

                # convert scoremaps into coordinates
                pred_hw_refine = [argmax_2d(smap) for smap in scoremap_pred_refined]
                score_refine = [tf.reduce_max(smap, [1, 2]) for smap in scoremap_pred_refined]
                pred_uv_refine = [tf.stack([x[:, :, 1], x[:, :, 0]], -1) for x in pred_hw_refine]

                if self.net_config.refinement_merge_method == 'triang':
                    # calculate 3D coordinates by triangulating the observations
                    pred_3d_refine, pred_3d_vis_refine = self.triangulate(pred_uv_refine[-1], score_refine[-1], K_tf, M_tf)

                elif self.net_config.refinement_merge_method == 'unproject':
                    # calculate 3D coordinates by unprojecting with depth constant assumption
                    pred_3d_refine, pred_3d_vis_refine = self.unproject_2d_estimates(
                        xyz_pred_final,
                        pred_uv_refine[-1],
                        score_refine[-1],
                        K_tf,
                        M_tf
                    )

                else:
                    raise NotImplementedError

        return scorevol_pred_list, xyz_pred_list, score_list, scoremap_pred_refined, pred_uv_refine, pred_3d_refine, pred_3d_vis_refine, variables

    def unproject_2d_estimates(self, pred_xyz_v0, pred_uv, scores, K, M):
        """
        Given an initial estimate in 3D
        """
        # get depth in cam for each point
        pred_xyz_cam = trafoPoints(pred_xyz_v0, M)

        # unproject points using depth from former prediction
        pred_xyz_cam_v1 = unproject(pred_uv, K, pred_xyz_cam[:, :, 2])  # this is B, N, 3

        # transform back to world space
        pred_xyz_v1 = trafoPoints(pred_xyz_cam_v1, tf.linalg.inv(M))

        # calculate weighted average over candidates
        pred_xyz = tf.reduce_sum(tf.multiply(tf.expand_dims(scores, -1), pred_xyz_v1), 0) / tf.expand_dims(tf.reduce_sum(scores, 0), -1)
        pred_conf = tf.reduce_sum(scores, 0)
        return tf.expand_dims(pred_xyz, 0), tf.expand_dims(pred_conf, 0)

    def triangulate(self, *args):
        _, num_kp, _ = args[0].get_shape().as_list()
        points3d, vis3d = tf.py_func(_triangulate, args, [tf.float32, tf.float32], name='pyfunc_triangulate')
        points3d.set_shape([1, num_kp, 3])
        vis3d.set_shape([1, num_kp])
        return points3d, vis3d

    def _inference2D_refine(self, scoremap_pred, enc_skips, is_training=True):
        """ Refines MVnets prediction according to the information from a view. """
        def _upsample_and_predict(x):
            pred = slim.conv2d(x, num_outputs=self.net_config.num_kp,
                               kernel_size=1, stride=1, activation_fn=None, trainable=is_training)
            pred_fs = tf.image.resize_bilinear(pred, (self.net_config.img_size, self.net_config.img_size))
            return pred_fs, pred

        # act_fct = tf.nn.relu
        act_fct = lambda x: tf.nn.leaky_relu(x, alpha=0.1)

        conv = slim.conv2d

        with tf.variable_scope('RefineNet2D') as scope:
            scoremaps = list()

            # get skip connection
            x_skip = conv(enc_skips.pop(), num_outputs=64, kernel_size=1, stride=1,
                          activation_fn=act_fct, trainable=is_training)

            # refine on base level
            x = tf.concat([scoremap_pred, x_skip], -1)
            x = conv(x, num_outputs=128, kernel_size=1, stride=1, activation_fn=act_fct,
                           trainable=is_training)
            x = conv(x, num_outputs=128, kernel_size=3, stride=1, activation_fn=act_fct,
                           trainable=is_training)

            scoremap_pred_fs, scoremap_pred = _upsample_and_predict(x)
            scoremaps.append(scoremap_pred_fs)

            # decoder
            for i, c in enumerate([128, 64]):
                # upsample last features
                x_up = slim.conv2d_transpose(x, num_outputs=c, kernel_size=4, stride=2,
                                             activation_fn=act_fct, trainable=is_training)

                # get skip connection
                x_skip = enc_skips.pop()
                x_skip = conv(x_skip, num_outputs=c // 2, kernel_size=1, stride=1,
                              activation_fn=act_fct, trainable=is_training)

                # upsample last prediction
                s = x_up.get_shape().as_list()
                scoremap_pred = tf.image.resize_bilinear(scoremap_pred, (s[1], s[2]))

                # joint processing
                x = tf.concat([x_up, x_skip, scoremap_pred], -1)
                x = conv(x, num_outputs=c, kernel_size=1, stride=1, activation_fn=act_fct,
                         trainable=is_training)
                x = conv(x, num_outputs=c, kernel_size=3, stride=1, activation_fn=act_fct,
                         trainable=is_training)

                scoremap_pred_fs, scoremap_pred = _upsample_and_predict(x)
                scoremaps.append(scoremap_pred_fs)

            variables = tf.contrib.framework.get_variables(scope)
            return scoremaps, variables

    def _mapping_network(self, feature3d, trainable, reuse):
        """ Maps warped features + eye vector into some common representation. """
        with tf.variable_scope('Mapping', reuse=reuse) as scope:
            x = feature3d
            # x = slim.conv3d(x, 64, kernel_size=[1, 1, 1], trainable=trainable, activation_fn=tf.nn.relu)
            x = slim.conv3d(x, 128, kernel_size=[1, 1, 1], trainable=trainable, activation_fn=tf.nn.relu)

            variables = tf.contrib.framework.get_variables(scope)

            return x, variables

    def _inference2D(self, image, trainable=True):
        """ Inference part of the network. """
        with tf.variable_scope('PoseNet2D') as scope:
            scoremap_kp_list = list()
            skips = list()

            layers_per_block = [2, 2, 4, 4, 2]
            out_chan_list = [64, 128, 256, 512, 512]
            pool_list = [True, True, True, False, False]
            train_list = [trainable, trainable, trainable, trainable, trainable]
            # train_list = [False, False, False, trainable, trainable]

            # learn some feature representation, that describes the image content well
            x = image

            for block_id, (layer_num, chan_num, pool, train) in enumerate(zip(layers_per_block, out_chan_list, pool_list, train_list), 1):
                for layer_id in range(layer_num):
                    x = ops.conv_relu(x, 'conv%d_%d' % (block_id, layer_id + 1), kernel_size=3, stride=1,
                                      out_chan=chan_num, trainable=train)
                if pool:
                    x = ops.max_pool(x, 'pool%d' % block_id)
                    skips.append(x)
            encoding = ops.conv_relu(x, 'conv5_3', kernel_size=3, stride=1, out_chan=128, trainable=trainable)

            # use encoding to detect initial scoremaps kp
            x = ops.conv_relu(encoding, 'conv6_1', kernel_size=1, stride=1, out_chan=512, trainable=trainable)
            scoremap_kp = ops.conv(x, 'conv6_2', kernel_size=1, stride=1, out_chan=self.config.num_kp, trainable=trainable)
            scoremap_kp_list.append(scoremap_kp)

            # # iterate refinement part a couple of times
            # num_recurrent_units = 1  # 2 already leads to worse performance
            # layers_per_recurrent_unit = 5
            # for pass_id in range(num_recurrent_units):
            #     # keypoints
            #     x = tf.concat([scoremap_kp_list[-1], encoding], 3)
            #     for layer_id in range(layers_per_recurrent_unit):
            #         x = ops.conv_relu(x, 'conv%d_%d' % (pass_id + 7, layer_id + 1), kernel_size=7, stride=1,
            #                           out_chan=128, trainable=trainable)
            #     x = ops.conv_relu(x, 'conv%d_%d' % (pass_id + 7, layers_per_recurrent_unit + 1), kernel_size=1,
            #                       stride=1, out_chan=128, trainable=trainable)
            #     scoremap_kp = ops.conv(x, 'conv%d_%d' % (pass_id + 7, 7), kernel_size=1, stride=1, out_chan=self.config.num_kp, trainable=trainable)
            #
            #     scoremap_kp_list.append(scoremap_kp)

            variables = tf.contrib.framework.get_variables(scope)

            return encoding, skips, variables


    @staticmethod
    def _enc3D_step(x, chan, dim_red, is_training):
        if dim_red:
            # optional dimensional reduction
            x = slim.conv3d(x, chan, kernel_size=[1, 1, 1], trainable=is_training, activation_fn=tf.nn.relu)
        x = slim.conv3d(x, chan, kernel_size=[3, 3, 3], trainable=is_training, stride=2, activation_fn=tf.nn.relu)
        return x

    def _dec3D_stop(self, x, skip, scorevol, chan, num_pred_chan, kernel2fs, is_training):
        # upsample features one step
        x_up = slim.conv3d_transpose(x, chan,
                                     kernel_size=[4, 4, 4], trainable=is_training, stride=2, activation_fn=tf.nn.relu)
        if skip is not None:
            # dim. reduction of the skip features
            x_skip = slim.conv3d(skip, chan, kernel_size=[1, 1, 1], trainable=is_training, activation_fn=tf.nn.relu)
            x = tf.concat([x_up, x_skip], -1)

        else:
            # if there is no skip left
            x = x_up

        # process features on the current resolution
        x = slim.conv3d(x, chan, kernel_size=[3, 3, 3], trainable=is_training, activation_fn=tf.nn.relu)

        # upsample all the way to full scale to make a prediction based on the current features
        scorevol_delta = slim.conv3d_transpose(x, num_pred_chan, kernel_size=[kernel2fs, kernel2fs, kernel2fs],
                                               stride=kernel2fs//2, trainable=is_training, activation_fn=None)

        scorevol = scorevol_delta
        return x, scorevol

    def _inference3D(self, x, is_training):
        """ Inference part of the network.

            Per view we get a 46x46x128 encoding (and maybe a 46x46x3 eye map).
            We unproject into a hand centered volume of dimension 64, so input dim is:
                64x64x64x 8*128 = 64x64x64x 1024
        """

        with tf.variable_scope('PoseNet3D') as scope:
            num_chan = self.config.num_kp

            skips = list()
            scorevolumes = list()
            skips.append(None)  # this is needed for the final upsampling step

            # 3D encoder
            # chan_list = [64, 128, 128, 256]
            chan_list = [32, 64, 64, 64]
            for chan in chan_list:
                x = self._enc3D_step(x, chan,
                                     dim_red=True, is_training=is_training)  # voxel sizes: 32, 16, 8, 4
                skips.append(x)
            skips.pop()  # the last one is of no use

            # bottleneck in the middle
            x = slim.conv3d(x, 64, kernel_size=[1, 1, 1], trainable=is_training,activation_fn=tf.nn.relu)

            # make initial guess of the scorevolume
            scorevol = slim.conv3d_transpose(x, num_chan, kernel_size=[32, 32, 32], trainable=is_training, stride=16, activation_fn=None)
            scorevolumes.append(scorevol)

            # 3D decoder
            kernels = [16, 8, 4]
            # chan_list = [64, 64, 64]
            chan_list = [32, 32, 32]
            for chan, kernel in zip(chan_list, kernels):
                x, scorevol = self._dec3D_stop(x, skips.pop(), scorevol, chan, num_chan, kernel, is_training)
                scorevolumes.append(scorevol)

            # final decoder step
            x = slim.conv3d_transpose(x, 64, kernel_size=[4, 4, 4], trainable=is_training, stride=2, activation_fn=tf.nn.relu)
            scorevol_delta = slim.conv3d(x, num_chan, kernel_size=[1, 1, 1], trainable=is_training, activation_fn=None)
            scorevol = scorevol_delta
            scorevolumes.append(scorevol)

            variables = tf.contrib.framework.get_variables(scope)

            if self.net_config.use_softargmax:
                xyz_vox_list = [softargmax3D(svol, output_vox_space=True) for svol in scorevolumes]
                score_list = [tf.reduce_mean(svol, [1, 2, 3]) for svol in scorevolumes]
            else:
                xyz_vox_list = [argmax_3d(svol) for svol in scorevolumes]
                score_list = [tf.reduce_max(svol, [1, 2, 3]) for svol in scorevolumes]

            return scorevolumes, xyz_vox_list, score_list, variables

    def _vox2metric(self, coords_vox, root_metric):
        """ Converts voxel coordinates in metric coordinates. """
        coords_metric = self.config.voxel_resolution / (self.config.voxel_dim-1) * (coords_vox - (self.config.voxel_dim-1) / 2.0) + root_metric
        return coords_metric

    def load_pretrained_weights(self, session):
        """ Load pretrained weights. """
        pass

    def get_losses(self, global_step):
        """ Define network losses. """
        losses = list()
        if self.net_config.use_softargmax:
            losses.append( Loss(target=target_t.generator, type=loss_t.l2,
                                weight=0.1, flag=None,
                                gt=data_t.xyz_nobatch, pred=data_t.pred_xyz) )  # loss for 3D network
        else:
            losses.append( Loss(target=target_t.generator, type=loss_t.l2_scorevolume,
                                weight=10.0, flag=None,
                                gt=data_t.xyz_vox_nobatch, pred=data_t.pred_scorevol) )  # loss for 3D network

        # flag scheduler
        self.flag_ref = 1.0 - tf.train.exponential_decay(1.0, global_step,
                                                         decay_steps=self.net_config.flag_step,
                                                         decay_rate=0.0,
                                                         staircase=True)
        self._summaries.append(
            tf.summary.scalar('flag_ref', self.flag_ref)
        )

        if self.net_config.use_2drefinement_net:
            losses.append( Loss(target=target_t.generator, type=loss_t.l2_scoremap,
                                weight=5.0, flag=self.flag_ref,
                                gt=data_t.uv_merged, pred=data_t.pred_scoremap) )  # aux loss for 2D networks

        if self.net_config.semisup:
            # Loss on length of bones
            losses.append(Loss(target=target_t.generator, type=loss_t.limb_length,
                               weight=0.1, flag=self.flag_ref,
                               gt=None, pred=data_t.pred_xyz))

            # loss on angle between neighboring limbs
            losses.append(Loss(target=target_t.generator, type=loss_t.limb_angles,
                               weight=0.1, flag=self.flag_ref,
                               gt=None, pred=data_t.pred_xyz))  # loss for 3D network

        # merge summaries
        if len(self._summaries) > 0:
            self._summaries = tf.summary.merge(self._summaries)

        return losses

    def get_tasks(self):
        """ Retrieve all kind of tasks this network would like to run. """
        tasks = list()

        # debugging
        # tasks.append( Task(name='debug', freq=1, offset=0,
        #                    pre_func=self.prefunc_debug, post_func=self.postfunc_debug) )

        # optimizer
        tasks.append( Task(name='optimizer', freq=1, offset=0, pre_func=self.prefunc_train, post_func=None) )

        # summaries that run always
        tasks.append( Task(name='summaries_always', freq=1, offset=0,
                           pre_func=self.prefunc_always_summaries,
                           post_func=self.postfunc_always_summaries) )

        # # visual summary that run sometimes
        # self.merged_vis = tf.placeholder(shape=(self.net_config.save_sample_num,
        #                                         self.config.img_size,
        #                                         self.config.img_size * 3, 3),
        #                             name='merged_vis_feed', dtype=tf.uint8)
        # self.merged_vis_sum = tf.summary.image('merged_vis', self.merged_vis, self.net_config.save_sample_num)
        # tasks.append( Task(name='summaries_save_sample',
        #                    freq=self.net_config.save_sample_freq, offset=0,
        #                    pre_func=self.prefunc_save_samples,
        #                    post_func=self.postfunc_save_samples) )

        # evaluation
        self.eval2d_epe = tf.placeholder(dtype=np.float32, shape=(), name='eval2d_epe_v')
        self.eval2d_auc = tf.placeholder(dtype=np.float32, shape=(), name='eval2d_auc_v')
        self.eval2d_epe_aux = tf.placeholder(dtype=np.float32, shape=(), name='eval2d_epe_aux_v')
        self.eval2d_auc_aux = tf.placeholder(dtype=np.float32, shape=(), name='eval2d_auc_aux_v')
        self.eval3d_epe = tf.placeholder(dtype=np.float32, shape=(), name='eval3d_epe_v')
        self.eval3d_auc = tf.placeholder(dtype=np.float32, shape=(), name='eval3d_auc_v')
        self.eval_vis_img = tf.placeholder(shape=(self.net_config.save_sample_num,
                                                self.config.img_size,
                                                self.config.img_size * 3, 3),
                                    name='eval_vis_img_feed', dtype=tf.uint8)
        self.eval_summary = tf.summary.merge([tf.summary.scalar('eval2d_epe', self.eval2d_epe),
                                              tf.summary.scalar('eval2d_auc', self.eval2d_auc),
                                              tf.summary.scalar('eval2d_epe_aux', self.eval2d_epe_aux),
                                              tf.summary.scalar('eval2d_auc_aux', self.eval2d_auc_aux),
                                              tf.summary.scalar('eval3d_epe', self.eval3d_epe),
                                              tf.summary.scalar('eval3d_auc', self.eval3d_auc),
                                              tf.summary.image('eval_vis_img', self.eval_vis_img,
                                                               self.net_config.save_sample_num)])
        tasks.append( Task(name='eval_test',
                           freq=self.net_config.eval_freq, offset=0,
                           pre_func=None,
                           post_func=self.postfunc_evaluate_test) )

        # evaluation on TRAIN set
        self.eval2d_epe_train = tf.placeholder(dtype=np.float32, shape=(), name='eval2d_epe_train_v')
        self.eval2d_auc_train = tf.placeholder(dtype=np.float32, shape=(), name='eval2d_auc_train_v')
        self.eval2d_epe_train_aux = tf.placeholder(dtype=np.float32, shape=(), name='eval2d_epe_train_aux_v')
        self.eval2d_auc_train_aux = tf.placeholder(dtype=np.float32, shape=(), name='eval2d_auc_train_aux_v')
        self.eval3d_epe_train = tf.placeholder(dtype=np.float32, shape=(), name='eval3d_epe_train_v')
        self.eval3d_auc_train = tf.placeholder(dtype=np.float32, shape=(), name='eval3d_auc_train_v')
        self.eval_vis_img_train = tf.placeholder(shape=(self.net_config.save_sample_num,
                                                self.config.img_size,
                                                self.config.img_size * 3, 3),
                                    name='eval_vis_img_feed_train', dtype=tf.uint8)
        self.eval_summary_train = tf.summary.merge([tf.summary.scalar('eval2d_train_epe', self.eval2d_epe_train),
                                                    tf.summary.scalar('eval2d_train_auc', self.eval2d_auc_train),
                                                    tf.summary.scalar('eval2d_train_epe_aux', self.eval2d_epe_train_aux),
                                                    tf.summary.scalar('eval2d_train_auc_aux', self.eval2d_auc_train_aux),
                                                    tf.summary.scalar('eval3d_train_epe', self.eval3d_epe_train),
                                                    tf.summary.scalar('eval3d_train_auc', self.eval3d_auc_train),
                                                    tf.summary.image('eval_vis_img_train', self.eval_vis_img_train,
                                                                     self.net_config.save_sample_num)])
        tasks.append( Task(name='eval_train',
                           freq=self.net_config.eval_freq, offset=0,
                           pre_func=None,
                           post_func=self.postfunc_evaluate_train) )
        return tasks

    def prefunc_save_samples(self, trainer):
        """ Define what to do with retrieved data. """
        # these lines say how to direct the dataflow numpy arrays 'values' to placeholders
        trainer.feeds[trainer.inputs[data_t.image]] = trainer.values['train_df'][data_t.image]
        trainer.feeds[trainer.inputs[data_t.K]] = trainer.values['train_df'][data_t.K]
        trainer.feeds[trainer.inputs[data_t.M]] = trainer.values['train_df'][data_t.M]
        trainer.feeds[trainer.ground_truths[data_t.xyz_nobatch]] = trainer.values['train_df'][data_t.xyz_nobatch]
        trainer.feeds[trainer.ground_truths[data_t.uv_merged]] = trainer.values['train_df'][data_t.uv_merged]

        trainer.fetches[data_t.image] = trainer.inputs[data_t.image]
        trainer.fetches[data_t.K] = trainer.inputs[data_t.K]
        trainer.fetches[data_t.M] = trainer.inputs[data_t.M]
        trainer.fetches[data_t.uv_merged] = trainer.ground_truths[data_t.uv_merged]
        trainer.fetches[data_t.xyz_nobatch] = trainer.ground_truths[data_t.xyz_nobatch]

        trainer.fetches[data_t.pred_uv_final] = trainer.predictions[data_t.pred_uv_final]
        trainer.fetches[data_t.pred_uv_refine] = trainer.predictions[data_t.pred_uv_refine]

    def postfunc_save_samples(self, trainer):
        """ Adds a visual sample to the summary writer. """
        import numpy as np
        from utils.plot_util import draw_skel

        tmp = list()
        for bid, (img, uv_gt, uv_pred, pred_uv_refine) in enumerate(zip(trainer.fetches_v[data_t.image],
                                                                        trainer.fetches_v[data_t.uv_merged],  # ground truth
                                                                        trainer.fetches_v[data_t.pred_uv_final])):  # projection

            img_rgb = ((img + 0.5) * 255).round().astype(np.uint8)[:, :, ::-1]
            img_p1 = img_rgb.copy()
            if self.net_config.use_2drefinement_net:
                img_p1 = draw_skel(img_p1, self.net_config.model,
                                   trainer.fetches_v[data_t.pred_uv_refine][-1][bid],
                                   order='uv')  # this is estimated from the single views

            img_p2 = draw_skel(img_rgb.copy(), self.net_config.model, uv_pred, order='uv')
            img_gt = draw_skel(img_rgb.copy(), self.net_config.model, uv_gt, order='uv')
            tmp.append(np.concatenate([img_p1, img_p2, img_gt], 1))

            if len(tmp) == trainer.config.save_sample_num:
                break

        summary_v = trainer.session.run(self.merged_vis_sum, {self.merged_vis: np.stack(tmp)})
        trainer.summary_writer.add_summary(summary_v, trainer.global_step_v)
        trainer.summary_writer.flush()
        print('Saved some samples.')

    def postfunc_evaluate_test(self, trainer):
        """ Run evaluation. """
        from colored import stylize, fg
        print(stylize('Running evaluation on ', fg('cyan')), stylize(' TEST', fg('red')))
        self._evaluate_on_set(trainer, trainer.dataflows['test_df'],
                              self.eval_summary,
                              self.eval2d_epe, self.eval2d_auc,
                              self.eval2d_epe_aux, self.eval2d_auc_aux,
                              self.eval3d_epe, self.eval3d_auc,
                              self.eval_vis_img, is_eval=True)

    def postfunc_evaluate_train(self, trainer):
        """ Run evaluation. """
        from colored import stylize, fg
        print(stylize('Running evaluation on ', fg('cyan')), stylize(' TRAIN', fg('red')))
        self._evaluate_on_set(trainer, trainer.dataflows['train_df'],
                              self.eval_summary_train,
                              self.eval2d_epe_train, self.eval2d_auc_train,
                              self.eval2d_epe_train_aux, self.eval2d_auc_train_aux,
                              self.eval3d_epe_train, self.eval3d_auc_train,
                              self.eval_vis_img_train, is_eval=False)

    def _evaluate_on_set(self, trainer, dataflow, summary,
                         epe2d_tf, auc2d_tf,
                         epe2d_tf_aux, auc2d_tf_aux,
                         epe3d_tf, auc3d_tf,
                         img_vis, is_eval):
        """ Run evaluation. """
        from utils.eval_util import EvalUtil
        from colored import stylize, fg
        from utils.plot_util import draw_skel, draw_text

        df2dict, df = dataflow

        # EVAL on TRAIN SET
        eval3d = EvalUtil()
        eval3d_refine = EvalUtil()
        eval2d = EvalUtil()
        eval2d_aux = EvalUtil()
        img_dump = list()
        for i in tqdm(range(trainer.config.eval_steps)):
            data = df2dict(next(df.get_data()))
            feed = {trainer.inputs[k]: data[k] for k in self.needed_inputs}

            #setup fetch
            fetches = [
                trainer.predictions[data_t.pred_xyz_final],
                trainer.predictions[data_t.pred_uv_final],
            ]

            # if refinement case show these, otherwise show auxiliary keypoint predictions

            if self.net_config.use_2drefinement_net:
                fetches.append(trainer.predictions[data_t.pred_uv_refine][-1])
                fetches.append(trainer.predictions[data_t.pred_xyz_refine])
                fetches.append(trainer.predictions[data_t.pred_vis3d_refine])

            # forward pass
            fetches_v = trainer.session.run(fetches, feed)
            fetches_v = fetches_v[::-1]  #reverse

            # extract values
            kp3d_pred = fetches_v.pop()
            kp2d_pred = fetches_v.pop()

            if self.net_config.use_2drefinement_net:
                kp2d_pred_alt = fetches_v.pop()
                pred_xyz_refine = fetches_v.pop()
                pred_vis3d_refine = fetches_v.pop()

            img = data[data_t.image]
            xyz_gt_v = data[data_t.xyz_nobatch]
            kp_vis_gt_v = data[data_t.vis_nobatch]
            kp_uv_gt_v = data[data_t.uv_merged]

            eval3d.feed(xyz_gt_v[0], kp_vis_gt_v[0], kp3d_pred[0])

            if self.net_config.use_2drefinement_net:
                eval3d_refine.feed(xyz_gt_v[0], kp_vis_gt_v[0], pred_xyz_refine[0])

            for bid in range(trainer.config.batch_size):
                eval2d.feed(kp_uv_gt_v[bid], kp_vis_gt_v[0], kp2d_pred[bid])
                eval2d_aux.feed(kp_uv_gt_v[bid], kp_vis_gt_v[0], kp2d_pred_alt[bid])

                if len(img_dump) < self.net_config.save_sample_num:
                    # assemble image
                    img_rgb = ((img[bid] + 0.5) * 255).round().astype(np.uint8)[:, :, ::-1]
                    img_p1 = img_rgb.copy()
                    if self.net_config.use_2drefinement_net:
                        img_p1 = draw_skel(img_p1, self.config.model, kp2d_pred_alt[bid], order='uv')  # this is estimated from the single view
                        img_p1 = draw_text(img_p1, 'ref')

                    img_p2 = draw_skel(img_rgb.copy(), self.config.model, kp2d_pred[bid], order='uv')
                    img_p2 = draw_text(img_p2, 'proj')
                    img_gt = draw_skel(img_rgb.copy(), self.config.model, kp_uv_gt_v[bid], order='uv')
                    img_gt = draw_text(img_gt, 'gt')
                    img_dump.append(np.concatenate([img_p1, img_p2, img_gt], 1))

        # get eval results and write to log
        mean2d, median2d, auc2d, _, _ = eval2d.get_measures(0.0, 100, 100)
        print(stylize('Evaluation 2D results:', fg('cyan')))
        print(stylize('auc=%.3f, mean_kp2d_avg=%.2f, median_kp2d_avg=%.2f' % (auc2d, mean2d, median2d), fg('green')))

        mean2d_aux, median2d_aux, auc2d_aux, _, _ = eval2d_aux.get_measures(0.0, 100, 100)
        if self.net_config.use_2drefinement_net:
            print(stylize('Evaluation 2D results (refinement):', fg('cyan')))
        else:
            print(stylize('Evaluation 2D results (aux):', fg('cyan')))
        print(stylize('auc=%.3f, mean_kp2d_avg=%.2f, median_kp2d_avg=%.2f' % (auc2d_aux, mean2d_aux, median2d_aux), fg('green')))

        if self.net_config.use_2drefinement_net:
            mean3d, median3d, auc3d, _, _ = eval3d_refine.get_measures(0.0, 0.05, 100)
            print(stylize('Evaluation 3D (refine) results:', fg('cyan')))
            print(stylize('auc=%.3f, mean_kp3d_avg=%.2f cm, median_kp3d_avg=%.2f  cm' % (auc3d,
                                                                                   mean3d*100.0,
                                                                                   median3d*100.0), fg('green')))

        mean3d, median3d, auc3d, _, _ = eval3d.get_measures(0.0, 0.05, 100)
        print(stylize('Evaluation 3D results:', fg('cyan')))
        print(stylize('auc=%.3f, mean_kp3d_avg=%.2f cm, median_kp3d_avg=%.2f  cm' % (auc3d,
                                                                               mean3d*100.0,
                                                                               median3d*100.0), fg('green')))

        if self.config.use_early_stopping and is_eval:
            trainer.es_util.feed(mean3d)

        eval_summary_train_v = trainer.session.run(summary,
                                        {epe2d_tf: np.clip(median2d, 0.0, 200.0), auc2d_tf: auc2d,
                                         epe2d_tf_aux: np.clip(median2d_aux, 0.0, 200.0), auc2d_tf_aux: auc2d_aux,
                                         epe3d_tf: np.clip(median3d, 0.0, 1.0), auc3d_tf: auc3d,
                                         img_vis: np.stack(img_dump)})
        trainer.summary_writer.add_summary(eval_summary_train_v, trainer.global_step_v)




