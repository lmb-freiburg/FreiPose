from __future__ import print_function, unicode_literals
from collections import OrderedDict, defaultdict
import os
import time
import datetime
import tensorflow as tf
from tensorflow.python.client import timeline
from tensorflow.python import pywrap_tensorflow
from tensorflow.core.protobuf import config_pb2
from colored import stylize, fg

from pose_network.core.Losses import loss_factory
from pose_network.core.Placeholders import ph_factory
from datareader.DataflowFactory import create_dataflow
from pose_network.nets.NetworkFactory import create_networks
from pose_network.nets.commons import *
from .Types import *

from utils.general_util import EarlyStoppingUtil


class Trainer(object):
    def __init__(self, config):
        self.config = config

        self.session = None
        self.global_step = None
        self.global_step_v = None
        self.time_per_fp = None
        self.time_per_iter = None
        self._tmp_time_fp = None
        self._tmp_time_iter = None
        self._start = time.time()
        self.time_per_fp_ph = None

        self.saver = None
        self.summaries = list()
        self.grad_norm_summaries = list()
        self.summary_writer = None

        self.losses = None
        self.loss_parts = None
        self.dataflows = dict()
        self.inputs = dict()  # contains ph considered to be input
        self.predictions = dict()
        self.ground_truths = dict()

        self.fetches = OrderedDict()
        self.fetches_v = OrderedDict()
        self.feeds = dict()
        self.values = dict()

        self.es_util = None

    def _setup(self):
        """ Setup the graph. """
        # if a config file is specified we load config from there
        if self.config.config_file is not None:
            self._load_from_config_file()

        # create fp runtime
        self.time_per_fp_ph = tf.placeholder(dtype=tf.float32, shape=(), name='time_per_fp_v')
        self.summaries.append(tf.summary.scalar('time_per_fp', self.time_per_fp_ph))

        # create network object
        self.network_list = create_networks(self.config)

        # accumulate inputs and outputs
        needed_inputs, needed_ground_truths = list(), list()
        for net in self.network_list:
            needed_inputs.extend(net.needed_inputs)
            needed_ground_truths.extend(net.needed_ground_truths)

        # create input/ground truth placeholders
        self.inputs = ph_factory(self.config, needed_inputs)
        self.ground_truths = ph_factory(self.config, needed_ground_truths)

        # setup networks
        self.losses = dict()
        self.loss_parts = dict()
        self.predictions = dict()
        self.global_step = tf.Variable(0, trainable=False, name='global_step')
        for net in self.network_list:
            # call inference function
            pred, var = net.inference(self.inputs, self.predictions, is_training=True)
            assert not any([k in self.predictions.keys() for k in pred.keys()]), 'Overwriting a predictions. Might indicate an error.'
            self.predictions.update(pred)

            # setup losses
            losses = net.get_losses(self.global_step)  # this defines what losses there are
            self.losses.update( self._setup_loss(losses, self.predictions, self.ground_truths))  # actually calculates the losses

            # setup optimizers
            lr = net.setup_optimizer(self.global_step, self.losses)
            self.summaries.append(tf.summary.scalar('learning_rate', lr))

        # loss summaries
        for t, l in self.losses.items():
            self.summaries.append(tf.summary.scalar('loss_%s' % t, l))

        # start session
        config = tf.ConfigProto(allow_soft_placement=True)
        config.gpu_options.allow_growth = True
        self.session = tf.Session(config=config)
        self.session.run(tf.global_variables_initializer())

        # load pretrained weights
        for net in self.network_list:
            net.load_pretrained_weights(self.session)

        # load starting checkpoint
        if self.config.starting_checkpoint is not None:
            self._load_all_variables_from_snapshot(self.config.starting_checkpoint, discard_list=['global_step'])

        # build dataflows
        self.dataflows = dict()
        for name, flow_type, kwargs in self.config.dataflows:
            df2dict, build_dataflow = create_dataflow(name, flow_type)
            self.dataflows[name] = (df2dict, build_dataflow(**kwargs))

        # setup summary writer
        summary_path = '%s/' % self.config.job_name
        if not os.path.exists(summary_path):
            os.makedirs(summary_path)
        self.summary_writer = tf.summary.FileWriter(summary_path, self.session.graph)

        # setup checkpoint saver
        self.saver = tf.train.Saver(max_to_keep=1)

        # get all tasks from the networks
        self.task_list = list()
        for net in self.network_list:
            self.task_list.extend(net.get_tasks())

        # append summaries that were created here
        self.summaries = tf.summary.merge(self.summaries)
        self.task_list.append( Task(name='trainer_summaries_always',
                                    freq=1, offset=0,
                                    pre_func=self.prefunc_always_summaries,
                                    post_func=self.postfunc_always_summaries) )

        self.task_list.append( Task(name='print_loss',
                                    freq=self.config.loss_show_freq,
                                    offset=0,
                                    pre_func=self.prefunc_print_loss,
                                    post_func=self.postfunc_print_loss) )

        # append grad summaries that were created here
        self.grad_norm_summaries = tf.summary.merge(self.grad_norm_summaries)
        self.task_list.append( Task(name='trainer_grad_summaries',
                                    freq=self.config.grad_sum_freq,
                                    offset=0,
                                    pre_func=self.prefunc_grad_summaries,
                                    post_func=self.postfunc_grad_summaries) )

        # Check if in the checkpoint folder is already data (which indicates this is a resuming job)
        self._load_resuming_checkpoint()

        if self.config.use_early_stopping:
            self.es_util = EarlyStoppingUtil(self.config.es_steps, self.config.es_rel_impr)

    def run(self):
        """ Starts training routine. """
        # network setup
        self._setup()

        # info about job
        print(stylize('This is: ', fg('blue')), stylize('%s' % self.config.job_name, fg('green')))

        # time when we get close to the 24h limit
        time_train_end = time.time() + 60*60*23*3  # change to incerase the timelimit to 24*3 h
        resubmit = False

        # training loop
        self.global_step_v = self.session.run(self.global_step)
        while self.global_step_v < self.config.pose_train_steps:
            self.fetches, self.feeds = OrderedDict(), dict()

            # check if checkpoint should be saved
            self.tic(is_first=True)
            if self._should_run(self.config.save_model_freq) and self.global_step_v > 0:
                self._save_snapshot()
            self.toc('save snapshot')

            # pull new data out of the streams that are tagges as is_train
            self.tic()
            self.values = dict()
            for name, (df2dict, df) in self.dataflows.items():
                if 'train' not in name:
                    # we only deal with training dataflows here
                    continue
                assert name not in self.values.keys(), 'Duplicate dataflow name. They need to be unique.'
                self.values[name] = dict()
                dp = next(df.get_data())
                new = df2dict(dp)
                assert not any([k in self.values[name].keys() for k in new.keys()]), 'Overwriting a key. Might indicate an error.'
                self.values[name].update(new)
            self.toc('get data')

            # call prefuncs of all tasks
            self.tic()
            for task in self.task_list:
                if task.pre_func is None or not self._should_run(task.freq, task.offset):
                    continue
                task.pre_func(self)  # this call usually modifies the feeds and fetches
            self.toc('prefunc')

            self.fetches['global_step'] = self.global_step

            self._timing_start()

            # run session
            self.tic()
            fetch_values = self.session.run(list(self.fetches.values()),
                                            feed_dict=self.feeds,
                                            options=config_pb2.RunOptions(report_tensor_allocations_upon_oom=True))
            self.toc('train')

            self._timing_stop()

            self.fetches_v = OrderedDict()
            for k, v in zip(self.fetches.keys(), fetch_values):
                self.fetches_v[k] = v

            # call postfuncs of all tasks
            self.tic()
            for task in self.task_list:
                if task.post_func is None or not self._should_run(task.freq, task.offset):
                    continue
                task.post_func(self)  # this uses the retrieved data
            self.toc('postfunc')

            self.global_step_v = self.fetches_v['global_step']

            # check 24h training limit
            if time.time() > time_train_end:
                resubmit = True
                break

            # check early stopping
            if self.config.use_early_stopping:
                if self.es_util.should_stop():
                    print('Hit ES criteria!')
                    break

        # save final snapshot
        self._save_snapshot()

        if resubmit:
            print('Training not finished but close to 24h limit: Resubmitting job.')
            self._resubmit_job()
            return
        else:
            print('Training finished.')

    def tic(self, is_first=False):
        if self.config.tictoc:
            if is_first:
                print('-------------------------------------------------------------')
            self._start = time.time()

    def toc(self, name):
        if self.config.tictoc:
            time_passed = time.time() - self._start
            print('%s: %.2f ms' % (name, time_passed*1000))

    def _timing_start(self):
        self._tmp_time_fp = time.time()

        if self.time_per_fp is None:
            self.feeds[self.time_per_fp_ph] = 0.0
        else:
            self.feeds[self.time_per_fp_ph] = self.time_per_fp

    def _timing_stop(self):
        if self.global_step_v == 0:
            # skip first iteration
            return

        f1 = 0.75
        f2 = 1.0 - f1
        if self.time_per_fp is None:
            self.time_per_fp = time.time() - self._tmp_time_fp
        else:
            self.time_per_fp = f1*self.time_per_fp + f2*(time.time() - self._tmp_time_fp)

        if self._tmp_time_iter is None:
            self._tmp_time_iter = time.time()
            return

        if self.time_per_iter is None:
            self.time_per_iter = time.time() - self._tmp_time_iter
        else:
            self.time_per_iter = f1*self.time_per_iter + f2*(time.time() - self._tmp_time_iter)
        self._tmp_time_iter = time.time()

    def _should_run(self, freq, offset=0):
        if freq > 0:
            return ((self.global_step_v - offset) % freq) == 0
        else:
            return False

    @staticmethod
    def _get_gradient_norm(x, trainable_vars):
        with tf.name_scope('get_gradient_norm'):
            gradients = tf.gradients(x, trainable_vars)
            gradient_norm = [tf.norm(g) for g in gradients if g is not None]
            if len(gradient_norm) > 0:
                return tf.add_n(gradient_norm) / float(len(gradient_norm))
            else:
                return 0.0

    def _setup_loss(self, losses, predictions, ground_truths):
        """ Creates losses according to what the network specified. """
        def _zero():
            return tf.constant(0.0)
        loss = defaultdict(_zero)  # dict of losses (different architecture parts may have seperate losses)
        for item in losses:
            this_loss = 0.0

            # create loss within its scope
            with tf.name_scope(item.type):
                this_loss += loss_factory(self.config, item, ground_truths, predictions)

            if item.flag is not None:
                # possibly mask this loss with a placeholder
                this_loss *= item.flag

            loss_key = '%s-%s-%s' % (item.type, item.gt, item.pred)
            assert loss_key not in self.loss_parts.keys(), 'Loss by same name does already exist.'
            self.loss_parts[loss_key] = this_loss
            self._loss_summary(item, this_loss)
            loss[item.target] += this_loss

        return loss

    def _loss_summary(self, item, this_loss):
        trainable_vars = tf.trainable_variables()  # list of trainables
        if self.config.grad_sum_freq == -1:
            this_grad = this_loss  # its never evaluated so we save the memory
        else:
            this_grad = self._get_gradient_norm(this_loss, trainable_vars)
        self.summaries.append(
            tf.summary.scalar('%s_%s_%s' % (item.type, item.gt, item.pred), this_loss)
        )
        self.grad_norm_summaries.append(
            tf.summary.scalar('grad_%s_%s_%s' % (item.type, item.gt, item.pred), this_grad)
        )

    def _load_from_config_file(self):
        """ Loads parameters from the config file and checks if there is a checkpoint available. """
        # load job from config
        self.config.load_from_file(self.config.config_file)

        # try to find a checkpoint file we probably want to resume from
        ckpt_path = '%s/ckpt/' % self.config.job_name
        last_file = tf.train.latest_checkpoint(ckpt_path)
        if last_file is not None:
            self.config.starting_checkpoint = last_file
            print('Auto detected checkpoints file: ', last_file)
            print('Training will start from there.')

    def _load_all_variables_from_snapshot(self, checkpoint_path, discard_list=None):
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
                num_disc += 1
        var_to_shape_map = dict(var_to_shape_map_new)
        print('Discarded %d items' % num_disc)
        # print('Vars in checkpoint', var_to_shape_map.keys(), len(var_to_shape_map))

        # get tensors to be filled
        var_to_values = dict()
        for name in var_to_shape_map.keys():
            value = reader.get_tensor(name)
            tensor = tf.get_default_graph().get_tensor_by_name(name+':0')
            if np.all(value.shape != tensor.shape):
                print('Shape mismatch when loading snapshot: %s value shape %s and tensor shape %s' % (name, value.shape, tensor.shape))
                continue
            var_to_values[name] = reader.get_tensor(name)

        init_op, init_feed = tf.contrib.framework.assign_from_values(var_to_values)
        self.session.run(init_op, init_feed)
        print(stylize('Initialized %d variables from %s.' % (len(var_to_shape_map), last_cpt), fg('green')))

    def _save_snapshot(self):
        ckpt_file = '%s/ckpt/model' % self.config.job_name
        # save model checkpoint
        ckpt_path = os.path.dirname(ckpt_file)
        if not os.path.exists(ckpt_path):
            os.makedirs(ckpt_path)
        self.saver.save(self.session, ckpt_file, global_step=self.global_step_v)
        print('Saved model checkpoint: %s-%d' % (ckpt_file, self.global_step_v))

    def _resubmit_job(self):
        """ Resubmits the job. """
        now = datetime.datetime.now()
        start_time = now + datetime.timedelta(minutes=5)
        print("ssh lmbtorque qsub -a %s %s/run_job.sh" % (start_time.strftime('%H%M'), os.getcwd()))
        os.system("ssh lmbtorque qsub -a %s %s/run_job.sh" % (start_time.strftime('%H%M'), os.getcwd()))

    def _load_resuming_checkpoint(self):
        """ Check if this is a resuming job. """
        checkpoint_path = os.path.join(os.getcwd(), 'ckpt')
        last_file = tf.train.latest_checkpoint(checkpoint_path)
        if last_file is not None:
            self.saver.restore(self.session, last_file)
            print('Resuming snapshot restored from: %s' % last_file)
        else:
            print('No snapshot to resume from was found.')

    def prefunc_always_summaries(self, _):
        """ Define what to do with retrieved data. """
        self.fetches['trainer_summaries_always'] = self.summaries

    def postfunc_always_summaries(self, _):
        """ Define what to do with retrieved data. """
        self.summary_writer.add_summary(self.fetches_v['trainer_summaries_always'], self.global_step_v)

    def prefunc_grad_summaries(self, _):
        """ Define what to do with retrieved data. """
        self.fetches['grad_norm_summaries'] = self.grad_norm_summaries

    def postfunc_grad_summaries(self, _):
        """ Define what to do with retrieved data. """
        self.summary_writer.add_summary(self.fetches_v['grad_norm_summaries'], self.global_step_v)

    def prefunc_print_loss(self, _):
        """ Define what to do with retrieved data. """
        for k, v in self.losses.items():
            self.fetches['loss_' + k] = v

    def postfunc_print_loss(self, _):
        """ Define what to do with retrieved data. """
        time_fp = self.time_per_fp if self.time_per_fp is not None else np.nan
        time_iter = self.time_per_iter if self.time_per_iter is not None else np.nan
        txt = 'Step: %d\t time per iter: %.2f\t time per fp: %.2f' % (self.global_step_v, time_iter, time_fp)

        for k, v in self.fetches_v.items():
            if 'loss_' in k:
                txt += '\t %s %g' % (k, v)
        print(txt)

