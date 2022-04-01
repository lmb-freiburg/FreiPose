""" Play around with kinematic constraints. """
from collections import defaultdict
import argparse
import numpy as np
import cv2
from tqdm import tqdm

from config.Model import Model
from pose_network.core.Types import *
from datareader.reader_labeled import build_dataflow, df2dict

from utils.plot_util import draw_skel
from utils.mpl_setup import plt_figure
from utils.StitchedImage import StitchedImage
import utils.CamLib as cl

def _get_coord(xyz, id_list):
    id_list = np.array(id_list).reshape([-1, 1])
    return np.mean(xyz[id_list], 0).squeeze()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Show labeled datasets.')
    parser.add_argument('model', type=str, help='Model definition file.')
    parser.add_argument('--set_name', type=str, default='train')
    parser.add_argument('--wait', action='store_true', help='Wait after each frame shown.')
    parser.add_argument('--window_size', type=int, default=1200, help='Window used for visualization.')
    parser.add_argument('--debug', action='store_true', help='Debug mode: Only subset of data is used.')
    args = parser.parse_args()

    model = Model(args.model)
    df = build_dataflow(model, [args.set_name], is_train=False,
                        threaded=False, single_sample=False)

    # data container to store all the body parameters we look at
    values = defaultdict(list)

    start = None
    for idx, dp in tqdm(enumerate(df.get_data()), desc='Accumulating stats', total=df.size()):
        if idx >= 100 and args.debug:
            break
        if idx >= df.size():
            break

        data = df2dict(dp)
        xyz = data[data_t.xyz_nobatch][0]
        vis = data[data_t.vis_nobatch]

        if np.sum(vis) < 12:
            # print('Not all vis', np.sum(vis))
            continue

        # limb lengths
        limbs = dict()
        for lid, (inds, _) in enumerate(model.limbs):
            pid, cid = inds
            values['limb%d' % lid].append(
                np.linalg.norm(_get_coord(xyz, cid) - _get_coord(xyz, pid), 2, -1)
            )

        # angles between dependent limbs
        for k, (lid1, lid2) in enumerate(model.data['kinematic_dep']):
            cid, pid = model.limbs[lid1][0]
            limb1 = _get_coord(xyz, cid) - _get_coord(xyz, pid)
            cid, pid = model.limbs[lid2][0]
            limb2 = _get_coord(xyz, cid) - _get_coord(xyz, pid)
            limb1 /= np.linalg.norm(limb1, 2)
            limb2 /= np.linalg.norm(limb2, 2)
            values['pair%d' % k].append(
                np.dot(limb1, limb2)
            )

    import matplotlib.pyplot as plt

    # # visualize distributions: Bone lengths
    # for lid, (inds, _) in enumerate(model.limbs):
    #     pid, cid = inds
    #     if type(pid) == int:
    #         pid = [pid]
    #     if type(cid) == int:
    #         cid = [cid]
    #
    #     n1 = [model.keypoints[x][0] for x in pid]
    #     n2 = [model.keypoints[x][0] for x in cid]
    #     print(n1, 'to', n2)
    #
    #     print('min, max: [%.3f, %.3f]' % (np.quantile(values['limb%d' % lid], 0.05),
    #                                       np.quantile(values['limb%d' % lid], 0.95)))
    #
    #     # fig, ax = plt.subplots(1, 1)
    #     # ax.hist(values['limb%d' % lid])
    #     # plt.show()

    # visualize distributions: Bone angles
    for k, v in values.items():
        if 'pair' not in k:
            continue

        print(k)

        print('min, max: [%.3f, %.3f]' % (np.quantile(v, 0.05),
                                          np.quantile(v, 0.95)))

        fig, ax = plt.subplots(1, 1)
        ax.hist(np.arccos(v)*180./np.pi)
        plt.show()

