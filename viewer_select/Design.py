import numpy as np
from pathlib import Path
import cv2
import os
from PyQt5.QtWidgets import *
import PyQt5.QtGui as QtGui
import PyQt5.QtCore as QtCore

from viewer_label.FrameView import FrameView

def shorten_path(file_path, num_parts=2):
    numparts = len(Path(file_path).parts)
    if numparts > num_parts:
        tmp = str(_shorten_path(file_path, num_parts))
        return '../' + str(tmp)
    return file_path


def _shorten_path(file_path, length):
    return Path(*Path(file_path).parts[-length:])

def create_kp_list(app, width, height):
    listwidget = QListWidget(app)
    listwidget.setSelectionMode(QAbstractItemView.ExtendedSelection)
    listwidget.resize(width, height)
    listwidget.setMinimumSize(width, height)
    listwidget.setMaximumSize(width, height)
    listwidget.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
    listwidget.setObjectName('Kp_list')
    app.kp_list = listwidget
    return listwidget


def create_frame_layout(app, width, height):
    # put everything in a vertical layout
    frame_layout_glob = QVBoxLayout()

    frame_layout = QVBoxLayout()
    frame_layout.setContentsMargins(5, 5, 5, 5)
    frame_layout.setSpacing(6)
    groupbox = QGroupBox('This Frame')
    groupbox.setLayout(frame_layout)

    pred_file = QLabel(app)
    pred_file.setText(shorten_path(app.pred_file, 1))
    pred_file.setToolTip(app.pred_file)
    frame_layout.addWidget(pred_file)

    #frame_button_layout = QHBoxLayout()
    #frame_button_layout.setContentsMargins(5, 5, 5, 5)
    #frame_button_layout.setSpacing(6)

    #frame_button_layout.addWidget(create_button(app, 'btn_del_frame', 'Delete all'), alignment=QtCore.Qt.AlignTop)
    #frame_button_layout.addWidget(create_button(app, 'btn_rec_frame', 'Recover'), alignment=QtCore.Qt.AlignTop)
    #frame_layout.addLayout(frame_button_layout)

    frame_layout.addWidget(create_button(app, 'btn_del_frame', 'Delete all'))
    frame_layout.addWidget(create_button(app, 'btn_rec_frame', 'Recover'))
    frame_layout.addWidget(create_kp_list(app, width, height))

    frame_layout_glob.addWidget(groupbox)
    frame_layout_glob.addStretch(1)

    return frame_layout_glob


def create_sel_file_layout(app, width, height):
    # put everything in a vertical layout
    file_layout_glob = QVBoxLayout()

    file_layout = QVBoxLayout()
    file_layout.setContentsMargins(5, 5, 5, 5)
    file_layout.setSpacing(6)
    groupbox = QGroupBox('Selected Frames')
    groupbox.setLayout(file_layout)

    listwidget = create_file_list(app, width, height)
    # for i in range(0, 100, 10):
    #     listwidget.addItem('%08d.png' % i)
    app.file_list_sel = listwidget
    file_layout.addWidget(listwidget)

    file_layout.addWidget(create_button(app, 'btn_sel_uni', 'Select uniform'))
    file_layout.addWidget(create_button(app, 'btn_sel_score', 'Select by score'))
    file_layout.addWidget(create_button(app, 'bn_sel_curr', 'Select current'))
    file_layout.addWidget(create_button(app, 'bn_unselect', 'Unselect marked'))
    file_layout.addWidget(create_button(app, 'bn_unselect_all', 'Unselect all'))
    file_layout.addLayout(create_line_edit_w_text(app, 'edit_sample', '#samples', 10))
    file_layout.addLayout(create_line_edit_w_text(app, 'edit_min_dist', 'distance', 10))

    file_layout_glob.addWidget(groupbox)
    file_layout_glob.addStretch(1)

    return file_layout_glob


def create_file_layout(app, width, height):
    # put everything in a vertical layout
    file_layout_glob = QVBoxLayout()

    file_layout = QVBoxLayout()
    file_layout.setContentsMargins(5, 5, 5, 5)
    file_layout.setSpacing(6)
    groupbox = QGroupBox('All Frames')
    groupbox.setLayout(file_layout)

    listwidget = create_file_list(app, width, height)
    for i in range(100):
        listwidget.addItem('%08d.png' % i)
    app.file_list = listwidget
    file_layout.addWidget(listwidget)

    file_layout.addWidget(create_button(app, 'btn_sort_id', 'Sort by id'))
    file_layout.addWidget(create_button(app, 'btn_sort_score', 'Sort by score'))

    file_layout_glob.addWidget(groupbox)
    file_layout_glob.addStretch(1)

    return file_layout_glob


def create_file_list(app, width, height):
    listwidget = QListWidget(app)
    listwidget.setMinimumSize(width, height)
    listwidget.setMaximumSize(width, height)
    listwidget.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
    # listwidget.setObjectName('file_list')
    return listwidget


def create_button(app, name, text):
    button = QPushButton(text, app)
    button.setObjectName(name)
    button.setMaximumWidth(150)
    return button


def create_line_edit_w_text(app, name, label_text, default_value, use_int_val=True, is_readonly=False):
    layout = QVBoxLayout()

    label = QLabel(app)
    label.setText(label_text)
    layout.addWidget(label)

    layout.addStretch(1)

    edit = QLineEdit(app)
    edit.setMaximumWidth(150)
    edit.setObjectName(name)
    edit.setText(str(default_value))
    if use_int_val:
        edit.setValidator(QtGui.QIntValidator(0, 1000, app))
    if is_readonly:
        edit.setReadOnly(True)
    layout.addWidget(edit)
    return layout


def set_icon(app):
    # draw
    img_np = np.ones((50, 50, 3), dtype=np.uint8)*255
    img_np = cv2.circle(img_np, (10, 10), radius=5, color=[0, 255, 0], thickness=2)
    img_np = cv2.circle(img_np, (40, 40), radius=5, color=[0, 0, 255], thickness=2)
    img_np = cv2.line(img_np, (10, 10), (40, 40), color=[255, 0, 0], thickness=1)

    # conversion
    qimage = QtGui.QImage(img_np.data, img_np.shape[1], img_np.shape[0], 3 * img_np.shape[1],
                       QtGui.QImage.Format_RGB888)
    icon = QtGui.QIcon()
    pixmap = QtGui.QPixmap.fromImage(qimage)
    icon.addPixmap(pixmap)

    # set
    app.setWindowIcon(QtGui.QIcon(icon))


def create_hist(app, width, height):
    hist_layout = QVBoxLayout()
    groupbox = QGroupBox('Score histogram')
    groupbox.setLayout(hist_layout)

    app.hist = QLabel(app)
    app.hist.resize(width, height)
    pix = QtGui.QPixmap('/home/zimmermc/projects/ALTool/results/dlc_cmp_cam_2dauc.png')
    app.hist.setPixmap(pix)
    hist_layout.addWidget(app.hist)
    return groupbox


def create_progressbar(app):
    pb_layout = QVBoxLayout()

    app.pb_label = QLabel(app)
    app.pb_label.setText('Writing progress:')
    app.pb_label.setVisible(True)
    pb_layout.addWidget(app.pb_label)

    app.pb = QProgressBar()
    app.pb.setMaximum(100)
    app.pb.setValue(0)
    app.pb.setVisible(True)
    pb_layout.addWidget(app.pb)
    return pb_layout


def create_output_menu(app):
    menu_layout = QVBoxLayout()
    menu_layout.setContentsMargins(5, 5, 5, 5)
    menu_layout.setSpacing(6)
    groupbox = QGroupBox('Output')
    groupbox.setLayout(menu_layout)

    # menu_layout.addLayout(create_line_edit_w_text(app, 'edit_output_path', 'Output path:', '',
    #                                               use_int_val=False, is_readonly=True))
    menu_layout.addLayout(create_progressbar(app))
    menu_layout.addWidget(create_button(app, 'btn_write', 'Write frames'))

    menu_layout.addStretch(1)
    return groupbox


def create_ui(app, config):
    app.setWindowTitle('Selection Tool')
    set_icon(app)

    main_layout = QHBoxLayout()
    main_layout.setContentsMargins(5, 5, 5, 5)
    main_layout.setSpacing(6)

    """ LEFT PART: Only frame view """

    s = config['frame_size']
    app.frame_view = FrameView(app, s[0], s[1],
                               config['keypoints'],
                               config['kp_radius'], config['kp_width'])
    main_layout.addWidget(app.frame_view, alignment=QtCore.Qt.AlignCenter)

    right_part = QVBoxLayout()
    right_part.setContentsMargins(5, 5, 5, 5)
    right_part.setSpacing(6)

    right_part_top = QHBoxLayout()
    right_part_top.setContentsMargins(5, 5, 5, 5)
    right_part_top.setSpacing(6)

    right_part_top1 = QVBoxLayout()
    right_part_top11 = QHBoxLayout()
    """ RIGHT PART FIRST COLUMN: KP list and frame point controlls. """
    right_part_top11.addLayout(create_frame_layout(app, 150, 200))

    """ RIGHT PART SECOND COLUMN: Frame list. """
    right_part_top11.addLayout(create_file_layout(app, 150, 200))
    right_part_top1.addLayout(right_part_top11)
    right_part_top1.addWidget(create_output_menu(app))
    right_part_top.addLayout(right_part_top1)

    """ RIGHT PART THIRD COLUMN: Selected frame list. """
    right_part_top.addLayout(create_sel_file_layout(app, 150, 200))

    right_part.addLayout(right_part_top)

    """ RIGHT PART BOTTOM: Histogram and Output"""
    # right_part_bottom = QHBoxLayout()
    # right_part_bottom.setContentsMargins(5, 5, 5, 5)
    # right_part_bottom.setSpacing(6)
    #
    # right_part_bottom.addWidget(create_hist(app, 300, 300), alignment=QtCore.Qt.AlignHCenter)
    # right_part_bottom.addWidget(create_output_menu(app))
    # right_part.addLayout(right_part_bottom)

    right_part.addWidget(create_hist(app, 50, 50), alignment=QtCore.Qt.AlignHCenter)
    main_layout.addLayout(right_part)
    main_layout.addStretch(1)
    main_layout.setSizeConstraint(QLayout.SetFixedSize)  # make it only use the space it really needs

    app.setLayout(main_layout)

