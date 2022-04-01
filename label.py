#!/usr/bin/env python3.6
import sys
import argparse
from PyQt5.QtWidgets import QApplication

from viewer_label.App import App
from config.Model import Model
from config.Param import Param

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Labeling tool.')
    parser.add_argument('model', type=str, help='Model definition file.')
    parser.add_argument('img_path', type=str, help='Path to the images.')
    parser.add_argument('--anno_file', type=str, default='anno.json', help='Path to the annotation data file, relative to img_path.')
    parser.add_argument('--calib_file', type=str, default='../M.json', help='Path to the annotation data file, relative to img_path. '
                                                                            'By default it is assumed the file is called M.json and '
                                                                            'located one directory down from img_path.')
    args = parser.parse_args()

    # load model data
    model = Model(args.model)
    param = Param()

    app = QApplication(sys.argv)
    ex = App(model, param, args.img_path, args.anno_file, args.calib_file)
    sys.exit(app.exec_())
