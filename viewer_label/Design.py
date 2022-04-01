import numpy as np
import cv2
import os
from PyQt5.QtWidgets import *
import PyQt5.QtGui as QtGui
import PyQt5.QtCore as QtCore

from .FrameView import FrameView
from .ExampleView import ExampleView


def create_kp_list(app, width, height):
    listwidget = QListWidget(app)
    listwidget.setSelectionMode(QAbstractItemView.ExtendedSelection)
    listwidget.resize(1150, 400)
    listwidget.setMinimumSize(width, height)
    listwidget.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
    listwidget.setObjectName('Kp_list')
    app.kp_list = listwidget
    return listwidget


def create_file_layout(app, width, height):
    # put everything in a vertical layout
    file_layout = QVBoxLayout()
    file_layout.setContentsMargins(11, 11, 11, 11)
    file_layout.setSpacing(6)

    file_layout_sub = QHBoxLayout()
    file_layout_sub.setContentsMargins(11, 11, 11, 11)
    file_layout_sub.setSpacing(6)
    file_layout_sub.addWidget(create_button(app, 'btn_prev', 'Prev'))
    file_layout_sub.addWidget(create_button(app, 'btn_next', 'Next'))
    file_layout.addLayout(file_layout_sub)

    listwidget = create_file_list(app, width, height)
    for i in range(100):
        listwidget.addItem('%08d.png' % i)
    app.file_list = listwidget
    file_layout.addWidget(listwidget)

    app.progress_label = QLabel(app)
    file_layout.addWidget(app.progress_label)
    app.base_path_label = QLabel(app)
    file_layout.addWidget(app.base_path_label)
    return file_layout


def create_file_list(app, width, height):
    listwidget = QListWidget(app)
    listwidget.setMinimumSize(width, height)
    listwidget.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
    listwidget.setObjectName('file_list')
    return listwidget


def create_button(app, name, text):
    button = QPushButton(text, app)
    button.setObjectName(name)
    return button


def create_checkbox(app, name, text, init_state):
    box = QCheckBox(text, app)
    box.setObjectName(name)
    box.setText(text)
    box.setChecked(init_state)
    return box


def create_line_edit(app, size, init_txt):
    this = QLineEdit(app)
    this.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.Minimum)
    this.setMinimumWidth(size)
    this.setText(init_txt)
    return this


def create_graphics_view(app, width, height):
    example_view = QGraphicsView(app)
    example_view.setMinimumSize(width, height)
    example_view.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
    example_view.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
    example_view.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
    example_view.setAlignment(QtCore.Qt.AlignTop | QtCore.Qt.AlignLeft)

    example_scene = QGraphicsScene(example_view)
    example_scene.setBackgroundBrush(QtGui.QBrush(QtCore.Qt.black))
    example_view.setSceneRect(0, 0, width, height)
    example_view.setScene(example_scene)
    example_view.setRenderHints(QtGui.QPainter.Antialiasing | QtGui.QPainter.SmoothPixmapTransform)
    return example_view, example_scene


def create_slider_layout(app, config):
    s = config['frame_size']
    slider_layout = QHBoxLayout()
    slider_layout.setContentsMargins(11, 11, 11, 11)
    slider_layout.setSpacing(6)
    app.slider = QSlider(app)
    app.slider.setFocusPolicy(QtCore.Qt.NoFocus)
    app.slider.setOrientation(QtCore.Qt.Horizontal)
    app.slider.setMinimumSize(s[0]-200, 10)
    app.slider.setRange(0, len(app.data))
    slider_layout.addWidget(app.slider)
    slider_layout.addWidget(create_button(app, 'btn_prev', 'Prev'))
    slider_layout.addWidget(create_button(app, 'btn_next', 'Next'))
    return slider_layout


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


def create_ui(app, config):
    app.setWindowTitle('Annotation Tool')
    set_icon(app)

    main_layout = QHBoxLayout()
    main_layout.setContentsMargins(11, 11, 11, 11)
    main_layout.setSpacing(6)

    # main left part
    left_layout = QVBoxLayout()
    left_layout.setContentsMargins(11, 11, 11, 11)
    left_layout.setSpacing(6)

    s = config['example_size']
    app.example_view = ExampleView(app, s[0], s[1], config['keypoints'],
                                   config['ex_kp_radius'], config['ex_kp_width'])
    img_bgr  = cv2.imread(config['example_path'])
    img_rgb = img_bgr[:, :, ::-1].copy()
    app.example_view.set_image(img_rgb)
    left_layout.addWidget(app.example_view)

    s = config['frame_size']
    app.frame_view = FrameView(app, s[0], s[1],
                               config['keypoints'],
                               config['kp_radius'], config['kp_width'])
    left_layout.addWidget(app.frame_view)

    # # slider layout
    # slider_layout = create_slider_layout(app, config)
    # left_layout.addLayout(slider_layout)

    # align centered everything thats in the left layout
    left_layout.setAlignment(app.example_view, QtCore.Qt.AlignCenter)
    left_layout.setAlignment(app.frame_view, QtCore.Qt.AlignCenter)
    # left_layout.setAlignment(app.slider, QtCore.Qt.AlignCenter)

    left_layout.addStretch(1)
    main_layout.addLayout(left_layout)

    # main right part
    right_layout = QVBoxLayout()
    right_layout.setContentsMargins(11, 11, 11, 11)
    right_layout.setSpacing(6)
    right_layout.addWidget(create_kp_list(app, 200, 400))
    right_layout.addWidget(create_button(app, 'btn_del', 'Delete all'))
    right_layout.addWidget(create_button(app, 'btn_tri', 'Triangulate'))
    right_layout.addWidget(create_button(app, 'btn_save', 'Save'))
    right_layout.addWidget(create_button(app, 'btn_load', 'Load'))
    right_layout.addWidget(create_checkbox(app, 'ckbox_repr', 'Draw reprojections', True))
    right_layout.addWidget(create_checkbox(app, 'ckbox_center', 'Draw center', False))
    right_layout.addStretch(1)
    file_list_layout = create_file_layout(app, 200, 200)
    right_layout.addLayout(file_list_layout)

    main_layout.addLayout(right_layout)
    main_layout.addStretch(1)
    main_layout.setSizeConstraint(QLayout.SetFixedSize)  # make it only use the space it really needs

    app.setLayout(main_layout)

