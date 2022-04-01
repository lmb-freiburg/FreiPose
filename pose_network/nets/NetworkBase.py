from __future__ import print_function, unicode_literals
import tensorflow as tf
from tensorflow.python import pywrap_tensorflow
from tensorflow.python.ops import control_flow_ops

from pose_network.core.Types import *
from .commons import *


class NetworkBase(object):
    """ Contains stuff that most of the networks need. """
    def __init__(self, net_config):
        self.net_config = net_config

        self.needed_inputs = list()   #t.b. filled by subclass
        self.needed_ground_truths = list()  #t.b. filled by subclass

        self._pretrained_vars = None
        self._all_vars = list()

        # summary stuff
        self._summaries = list()
        self.train_pose = None
        self.merged_vis = None
        self.merged_vis_sum = None

        self.eval_vis_img = None
        self.eval2d_epe = None
        self.eval2d_auc = None
        self.eval2d_epe_aux = None
        self.eval2d_auc_aux = None
        self.eval3d_epe = None
        self.eval3d_auc = None
        self.eval_summary = None

        self.eval_vis_img_train = None
        self.eval2d_epe_train = None
        self.eval2d_auc_train = None
        self.eval2d_epe_train_aux = None
        self.eval2d_auc_train_aux = None
        self.eval3d_epe_train = None
        self.eval3d_auc_train = None
        self.eval_summary_train = None

    def load_pretrained_weights(self, session):
        """ Load pretrained weights. """
        restore_ckpt = tf.train.Saver(max_to_keep=1, var_list=self._pretrained_vars)
        restore_ckpt.restore(session, './data/resnet/resnet_v2_50.ckpt')
        print('Resnet weights loaded from checkpoint for initial init.')

    def setup_optimizer(self, global_step, losses):
        """ Get optimizers for this network. """
        # Learning rate scheduler
        learning_rate = tf.train.exponential_decay(self.net_config.learning_rate_base, global_step,
                                                   decay_steps=self.net_config.lr_decay_steps,
                                                   decay_rate=self.net_config.lr_decay_rate,
                                                   staircase=True)

        # batch norm update op
        bn_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        if bn_ops:
            losses[target_t.generator] = control_flow_ops.with_dependencies([tf.group(*bn_ops)],
                                                                            losses[target_t.generator])

        self.train_pose = tf.train.AdamOptimizer(
            learning_rate=learning_rate
        ).minimize(losses[target_t.generator],
                   global_step=global_step,
                   var_list=self._all_vars,
                   colocate_gradients_with_ops=True)
        return learning_rate

    def get_tasks(self):
        """ Retrieve all kind of tasks this network would like to run. """
        tasks = list()

        # debugging
        # tasks.append( Task(name='debug', freq=1, pre_func=self.prefunc_debug, post_func=self.postfunc_debug) )

        # optimizer
        tasks.append( Task(name='optimizer', freq=1, offset=0, pre_func=self.prefunc_train, post_func=None) )

        # summaries that run always
        tasks.append( Task(name='summaries_always', freq=1, offset=0,
                           pre_func=self.prefunc_always_summaries,
                           post_func=self.postfunc_always_summaries) )

        # visual summary that run sometimes
        self.merged_vis = tf.placeholder(shape=(self.net_config.save_sample_num,
                                                self.net_config.img_size,
                                                self.net_config.img_size * 2, 3),
                                    name='merged_vis_feed', dtype=tf.uint8)
        self.merged_vis_sum = tf.summary.image('merged_vis', self.merged_vis, self.net_config.save_sample_num)
        tasks.append( Task(name='summaries_save_sample',
                           freq=self.net_config.save_sample_freq, offset=0,
                           pre_func=self.prefunc_save_samples,
                           post_func=self.postfunc_save_samples) )

        # evaluation
        self.eval2d_epe = tf.placeholder(dtype=np.float32, shape=(), name='eval2d_epe_v')
        self.eval2d_auc = tf.placeholder(dtype=np.float32, shape=(), name='eval2d_auc_v')
        self.eval3d_epe = tf.placeholder(dtype=np.float32, shape=(), name='eval3d_epe_v')
        self.eval3d_auc = tf.placeholder(dtype=np.float32, shape=(), name='eval3d_auc_v')
        self.eval_summary = tf.summary.merge([tf.summary.scalar('eval2d_epe', self.eval2d_epe),
                                              tf.summary.scalar('eval2d_auc', self.eval2d_auc),
                                              tf.summary.scalar('eval3d_epe', self.eval3d_epe),
                                              tf.summary.scalar('eval3d_auc', self.eval3d_auc)])
        tasks.append( Task(name='eval_test', freq=self.net_config.eval_freq, offset=0,
                           pre_func=None,
                           post_func=self.postfunc_evaluate_test) )

        # evaluation on TRAIN set
        self.eval2d_epe_train = tf.placeholder(dtype=np.float32, shape=(), name='eval2d_epe_train_v')
        self.eval2d_auc_train = tf.placeholder(dtype=np.float32, shape=(), name='eval2d_auc_train_v')
        self.eval3d_epe_train = tf.placeholder(dtype=np.float32, shape=(), name='eval3d_epe_train_v')
        self.eval3d_auc_train = tf.placeholder(dtype=np.float32, shape=(), name='eval3d_auc_train_v')
        self.eval_summary_train = tf.summary.merge([tf.summary.scalar('eval2d_train_epe', self.eval2d_epe_train),
                                              tf.summary.scalar('eval2d_train_auc', self.eval2d_auc_train),
                                              tf.summary.scalar('eval3d_train_epe', self.eval3d_epe_train),
                                              tf.summary.scalar('eval3d_train_auc', self.eval3d_auc_train)])
        tasks.append( Task(name='eval_train', freq=self.net_config.eval_freq, offset=0,
                           pre_func=None,
                           post_func=self.postfunc_evaluate_train) )
        return tasks

    def prefunc_debug(self, trainer):
        """ Define what we need and what to provide in order to do this task. """
        trainer.fetches['my_xyz'] = trainer.ground_truths[data_t.xyz]
        trainer.fetches['my_xyz_pred'] = trainer.predictions[data_t.pred_xyz]

    def postfunc_debug(self, trainer):
        """ Define what we need and what to provide in order to do this task. """
        print('my_xyz', trainer.fetches_v['my_xyz'][0, :5, :])
        print('my_xyz_pred', trainer.fetches_v['my_xyz_pred'][0, :5, :])

    def prefunc_always_summaries(self, trainer):
        """ Define what to do with retrieved data. """
        if not type(self._summaries) == list:
            trainer.fetches['summaries_always'] = self._summaries

    def postfunc_always_summaries(self, trainer):
        """ Define what to do with retrieved data. """
        if not type(self._summaries) == list:
            trainer.summary_writer.add_summary(trainer.fetches_v['summaries_always'], trainer.global_step_v)

    def prefunc_train(self, trainer):
        """ Define what we need and what to provide in order to do this task. """
        trainer.fetches['train_pose_op'] = self.train_pose
        for t in self.needed_inputs:
            trainer.feeds[trainer.inputs[t]] = trainer.values['train_df'][t]

        for t in self.needed_ground_truths:
            trainer.feeds[trainer.ground_truths[t]] = trainer.values['train_df'][t]

    def prefunc_save_samples(self, trainer):
        """ Define what to do with retrieved data. """
        trainer.feeds[trainer.inputs[data_t.image]] = trainer.values['train_df'][data_t.image]
        trainer.feeds[trainer.inputs[data_t.K]] = trainer.values['train_df'][data_t.K]
        trainer.feeds[trainer.ground_truths[data_t.xyz]] = trainer.values['train_df'][data_t.xyz]

        trainer.fetches[data_t.image] = trainer.inputs[data_t.image]
        trainer.fetches[data_t.K] = trainer.inputs[data_t.K]
        trainer.fetches[data_t.xyz] = trainer.ground_truths[data_t.xyz]
        trainer.fetches[data_t.pred_xyz] = trainer.predictions[data_t.pred_xyz]

    def postfunc_save_samples(self, trainer):
        """ Adds a visual sample to the summary writer. """
        from utils.plot_util import draw_hand
        import utils.CamLib as cl
        import numpy as np

        tmp = list()
        for img, cam_int, xyz_pred, xyz_gt in zip(trainer.fetches_v[data_t.image],
                                                  trainer.fetches_v[data_t.K],
                                                  trainer.fetches_v[data_t.pred_xyz],
                                                  trainer.fetches_v[data_t.xyz]):
            # project
            uv_gt = cl.project(xyz_gt, cam_int)
            uv_pred = cl.project(xyz_pred, cam_int)

            img_rgb = ((img + 1.0) / 2.0 * 255).round().astype(np.uint8)
            img_p = draw_hand(img_rgb.copy(), uv_pred, order='uv')
            img_gt = draw_hand(img_rgb.copy(), uv_gt, order='uv')
            tmp.append(np.concatenate([img_p, img_gt], 1))

            # from utils.mpl_setup import plt_figure
            # plt, fig, axes = plt_figure(2)
            # axes[0].imshow(img_p)
            # axes[1].imshow(img_gt)
            # plt.show()

            if len(tmp) == trainer.config.save_sample_num:
                break

        summary_v = trainer.session.run(self.merged_vis_sum, {self.merged_vis: np.stack(tmp)})
        trainer.summary_writer.add_summary(summary_v, trainer.global_step_v)
        trainer.summary_writer.flush()
        print('Saved some samples.')

    def postfunc_evaluate_test(self, trainer):
        """ Run evaluation. """
        from colored import stylize, fg
        print(stylize('Running evaluation on ', fg('blue')), stylize(' TEST', fg('red')))
        self._evaluate_on_set(trainer, trainer.dataflows['test_df'],
                              self.eval_summary,
                              self.eval2d_epe, self.eval2d_auc,
                              self.eval3d_epe, self.eval3d_auc)

    def postfunc_evaluate_train(self, trainer):
        """ Run evaluation. """
        from colored import stylize, fg
        print(stylize('Running evaluation on ', fg('blue')), stylize(' TRAIN', fg('red')))
        self._evaluate_on_set(trainer, trainer.dataflows['train_df'],
                              self.eval_summary_train,
                              self.eval2d_epe_train, self.eval2d_auc_train,
                              self.eval3d_epe_train, self.eval3d_auc_train)

    def _evaluate_on_set(self, trainer, dataflow, summary, epe2d_tf, auc2d_tf, epe3d_tf, auc3d_tf):
        """ Run evaluation. """
        from utils.eval_util import EvalUtil
        from utils.ProgressBarEta import ProgressBarEta
        from colored import stylize, fg

        df2dict, df = dataflow

        # EVAL on TRAIN SET
        eval3d = EvalUtil()
        eval2d = EvalUtil()
        pb = ProgressBarEta(trainer.config.eval_steps)
        for _ in range(trainer.config.eval_steps):
            pb.update()

            data = df2dict(next(df.get_data()))
            feed = {trainer.inputs[k]: data[k] for k in self.needed_inputs}

            kp3d_pred, kp2d_pred = trainer.session.run([trainer.predictions[data_t.pred_xyz_final],
                                                        trainer.predictions[data_t.pred_uv_final]], feed)

            xyz_gt_v = data[data_t.xyz]
            kp_vis_gt_v = np.ones_like(xyz_gt_v[:, :, :1])
            kp_uv_gt_v = data[data_t.uv]

            for bid in range(trainer.config.batch_size):
                eval3d.feed(xyz_gt_v[bid], kp_vis_gt_v[bid], kp3d_pred[bid])
                eval2d.feed(kp_uv_gt_v[bid], kp_vis_gt_v[bid], kp2d_pred[bid])
        pb.finish()

        # get eval results and write to log
        mean2d, median2d, auc2d, _, _ = eval2d.get_measures(0.0, 100, 100)
        print(stylize('Evaluation 2D results:', fg('blue')))
        print(stylize('auc=%.3f, mean_kp2d_avg=%.2f, median_kp2d_avg=%.2f' % (auc2d, mean2d, median2d), fg('green')))
        mean3d, median3d, auc3d, _, _ = eval3d.get_measures(0.0, 0.05, 100)
        print(stylize('Evaluation 3D results:', fg('blue')))
        print(stylize('auc=%.3f, mean_kp3d_avg=%.2f cm, median_kp3d_avg=%.2f  cm' % (auc3d,
                                                                               mean3d*100.0,
                                                                               median3d*100.0), fg('green')))
        eval_summary_train_v = trainer.session.run(summary,
                                        {epe2d_tf: np.clip(median2d, 0.0, 200.0), auc2d_tf: auc2d,
                                         epe3d_tf: np.clip(median3d, 0.0, 1.0), auc3d_tf: auc3d})
        trainer.summary_writer.add_summary(eval_summary_train_v, trainer.global_step_v)

    def load_all_variables_from_snapshot(self, session, checkpoint_path, discard_list=None):
        """ Initializes certain tensors from a snapshot. """
        if discard_list is None:
            discard_list = list()

        last_cpt = tf.train.latest_checkpoint(checkpoint_path)
        assert last_cpt is not None, "Could not locate snapshot to load."
        reader = pywrap_tensorflow.NewCheckpointReader(last_cpt)
        var_to_shape_map = reader.get_variable_to_shape_map()  # var_to_shape_map

        # for name in var_to_shape_map.keys():
        #     print(name, reader.get_tensor(name).shape)

        # Remove everything from the discard list
        num_disc = 0
        var_to_shape_map_new = dict()
        for k, v in var_to_shape_map.items():
            good = True
            for dis_str in discard_list:
                if dis_str in k:
                    good = False

            if good:
                var_to_shape_map_new[k] = v
            else:
                # print('Discarded: ', k)
                num_disc += 1
        var_to_shape_map = dict(var_to_shape_map_new)
        print('Discarded %d items' % num_disc)
        # print('Vars in checkpoint', var_to_shape_map.keys(), len(var_to_shape_map))
        print('Good ones.')

        # for k, v in var_to_shape_map.items():
        #     print(k)

        # get tensors to be filled
        var_to_values = dict()
        for name in var_to_shape_map.keys():
            var_to_values[name] = reader.get_tensor(name)

        init_op, init_feed = tf.contrib.framework.assign_from_values(var_to_values)
        session.run(init_op, init_feed)
        print('Initialized %d variables from %s.' % (len(var_to_shape_map), last_cpt))

