from collections import defaultdict
import tikzplotlib as tpl
import argparse, os
import numpy as np
from prettytable import PrettyTable

from utils.mpl_setup import plt_figure
from utils.general_util import json_load


def to_mat(laser_data, num_time_steps):
    # create class label vector
    laser_pulse_width = 150
    labels = np.zeros((num_time_steps,))  # 0 dont know, 1 laser on (sure), -1 laser off (sure)
    laser_data = np.array(laser_data)  # frames where the laser was switched on
    laser_data_off = laser_data + laser_pulse_width # frames where the laser was switched on
    margin_off = 0
    margin_on_f = 0
    margin_on_b = 0
    for i in range(num_time_steps):
        for k in range(len(laser_data)):
            if i-margin_on_f >= laser_data[k] and i+margin_on_b <= laser_data_off[k]:
                labels[i] = 1.0

            if k == 0 and i+margin_off < laser_data[k]:
                labels[i] = -1.0

            if i-margin_off > laser_data_off[k]:
                if k < len(laser_data) - 1:
                    if i+margin_off < laser_data[k+1]:
                        labels[i] = -1.0
                else:
                    labels[i] = -1.0

    return labels


def _eval(this_mode, train_rat_list, eval_rat_list, margin=20):
    print('\nTYPE', this_mode)
    rows = ['train \ eval', 'fg acc', 'bg acc', 'avg acc', 'avg (weighted) acc', 'fg/bg cnt']
    # rows.extend(eval_rat_list)
    fg_acc_table = PrettyTable(rows)

    # aligned_pred = defaultdict(list)
    pred_all = list()
    for train_rat, eval_rat in zip(train_rat_list, eval_rat_list):
        # load data
        pred_file = os.path.join(
            BASE_PATH,
            '%s_%s' % (eval_rat, REC_NAME_DICT[this_mode][eval_rat]),
            'pred_full_t%s_e%s_.json' % (train_rat, eval_rat)
        )

        laser_data = os.path.join(
            os.path.dirname(pred_file),
            'times.json'
        )

        # assert os.path.exists(args.pred_cls), 'Pred file not found.'
        pred_data = json_load(pred_file)
        y_true, y_pred = np.array(pred_data['gt']), np.array(pred_data['pred'])

        assert os.path.exists(laser_data), 'Pred file not found.'
        laser_data = json_load(laser_data)['frames']

        # overall accuracy
        acc = np.mean(y_true == y_pred)
        # print('Overall accuracy', acc)

        # fg/bg accuracy
        m = y_true < 0.0
        bg_acc = np.mean(y_true[m] == y_pred[m])

        m = y_true > 0.0
        fg_acc = np.mean(y_true[m] == y_pred[m])
        # print('FG accuracy', acc)

        # accumulate aligned preditions
        for i in laser_data:
            # s, e = i, i+150
            s, e = i-margin, i+150+margin
            pred_all.append(y_pred[s:e])

        fg_cnt = np.sum(y_true > 0.0)
        bg_cnt = np.sum(y_true < 0.0)
        avg_acc = 0.5*(fg_acc + bg_acc)
        d = [train_rat, '%.1f' % (fg_acc*100.0), '%.1f' % (bg_acc*100.0),
             '%.1f' % (avg_acc*100.0), '%.1f' % (acc*100.0), '%d/%d' % (fg_cnt, bg_cnt)]
        fg_acc_table.add_row(d)

    pred_all = (np.mean(pred_all, 0) + 1.0) / 2.0
    pred_stim = (this_mode, np.arange(pred_all.shape[0]) - margin, pred_all)
    return pred_stim, fg_acc_table


if __name__ == '__main__':
    MODE_LIST = [
        '10Hz',
        '30Hz'
    ]
    RAT_LIST = [
        'Rat511',
        'Rat512',
        'Rat513'
    ]

    REC_NAME_DICT = {
        '10Hz': {
            'Rat511': '20191210_2',
            'Rat512': '20191210_2',
            'Rat513': '20191209_2'
        },
        '30Hz': {
            'Rat511': '20191209_2',
            'Rat512': '20191209_2',
            'Rat513': '20191210_2'
        }
    }

    MARGIN = 100
    BASE_PATH = '/misc/lmbraid18/zimmermc/datasets/RatTrack_Laser/Final_Trials/'

    # theses are: mode, train, eval, given_name
    JOBS = [
        ['10Hz', RAT_LIST, RAT_LIST],
        ['30Hz', RAT_LIST, RAT_LIST]
    ]
    factor = 1.0/30.0  # to scale frames into s

    pred_stim_all = list()
    for mode, train_rat_list, eval_rat_list in JOBS:
        pred_stim, fg_acc_table = _eval(mode, train_rat_list, eval_rat_list, MARGIN)
        print(fg_acc_table)
        pred_stim_all.append(pred_stim)

    # plot triggered
    plt, fig, axes = plt_figure(1)
    for mode, x, pred_all in pred_stim_all:
        # average a bit
        N = 7
        f = np.ones((N,))
        f /= np.sum(f)
        pred_q = np.convolve(pred_all, f, mode='SAME')[::N]
        x_q = x[::N]

        # axes[0].bar(x_q, pred_q, width=0.8 * N)
        axes[0].plot(x_q*factor, pred_q, label='Prediction average (%s)' % mode)

    t = np.arange(-MARGIN, 150+MARGIN)
    v = np.logical_and(t >=0, t<=150).astype(np.float32)
    # axes[0].plot(t, v, label='Laser')
    axes[0].plot(t*factor, v, label='Stimulation ')
    axes[0].set_xlabel('Time in s')
    axes[0].set_ylabel('Stimulus')
    # axes[0].set_xlim([-3, 150])
    # axes[0].set_ylim([0.0, 1.0])
    plt.legend()
    fig.set_size_inches(12, 12)
    fig.tight_layout()
    tpl.save('/home/zimmermc/projects/FreiPose/result_figures/laser_pred_stimulus_s.tikz')
    fig.savefig('/home/zimmermc/projects/FreiPose/result_figures/laser_pred_stimulus_s.png', dpi=100)
    # plt.show(block=False)
    plt.show()

