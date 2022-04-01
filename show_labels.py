import argparse
import numpy as np
import cv2

from config.Model import Model
from pose_network.core.Types import *
from datareader.reader_labeled import build_dataflow, df2dict

from utils.plot_util import draw_skel
from utils.mpl_setup import plt_figure
from utils.StitchedImage import StitchedImage
import utils.CamLib as cl


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Show labeled datasets.')
    parser.add_argument('model', type=str, help='Model definition file.')
    parser.add_argument('--set_name', type=str, default='train')
    parser.add_argument('--wait', action='store_true', help='Wait after each frame shown.')
    parser.add_argument('--window_size', type=int, default=1200, help='Window used for visualization.')
    args = parser.parse_args()

    model = Model(args.model)
    df = build_dataflow(model, [args.set_name], is_train=True,
                        threaded=False, single_sample=True)

    start = None
    for idx, dp in enumerate(df.get_data()):
        if idx >= df.size():
            break

        data = df2dict(dp)
        img_rgb = np.round((data[data_t.image]+0.5)*255.0).astype(np.uint8)[:, :, :, ::-1]
        num_cams = img_rgb.shape[0]

        img_list = list()
        for i in range(num_cams):
            xyz_cam = cl.trafo_coords(data[data_t.xyz_nobatch][0], data[data_t.M][i])
            uv = cl.project(xyz_cam, data[data_t.K][i])
            # I = draw_skel(img_rgb[i], model, data[data_t.uv_merged][i], data[data_t.vis_nobatch][0], order='uv')
            # I = draw_skel(img_rgb[i], model, data[data_t.uv][i], data[data_t.vis_nobatch][0], order='uv')
            I = draw_skel(img_rgb[i], model, uv, data[data_t.vis_nobatch][0], order='uv')
            img_list.append(I)
        xyz = data[data_t.xyz_nobatch][0]

        merge = StitchedImage(img_list, target_size=(int(0.8 * args.window_size), args.window_size))

        cv2.imshow('pose labeled', merge.image[:, :, ::-1])
        cv2.waitKey(0 if args.wait else 10)

