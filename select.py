#!/usr/bin/env python3.6
import sys, os
import argparse
from PyQt5.QtWidgets import QApplication

from config.Model import Model
from config.Param import Param
from utils.general_util import json_load, parse_file_name
from viewer_select.App import App

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Show labeled datasets.')
    parser.add_argument('model', type=str, help='Model definition file.')
    parser.add_argument('file', type=str, help='Either a prediction (*.json) file or '
                                               ' one of the video files.')
    parser.add_argument('--calib_file', type=str, default='M.json', help='Relative path to the conlibration file.')

    parser.add_argument('--run_wildcard', type=str, default='run%03d', help='How to tell the run id'
                                                                            ' from a given file name.')
    parser.add_argument('--rec_fmt', type=str, default='%s_cam%d.avi', help='Re for the recording.')
    parser.add_argument('--max_cams', type=int, default=64, help='Maximal number of cams we search for.')
    args = parser.parse_args()

    predictions = None
    if args.file.endswith('.json'):
        # load predictions
        predictions = json_load(args.file, verbose=True)

    # extract run from file name
    base_path, run_id = parse_file_name(args.file, args.run_wildcard)

    # Parse videos
    video_list = [(i, os.path.join(base_path, args.rec_fmt % (args.run_wildcard % run_id, i))) for i in range(args.max_cams)]
    video_list = [v for v in video_list if os.path.exists(v[1])]
    cam_range, video_list = [v[0] for v in video_list], [v[1] for v in video_list]

    # load model data
    model = Model(args.model)
    param = Param()

    app = QApplication(sys.argv)
    ex = App(args.file, model, param,
             predictions, video_list, cam_range, args.calib_file)
    sys.exit(app.exec_())
