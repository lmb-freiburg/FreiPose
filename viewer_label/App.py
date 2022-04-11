from collections import defaultdict
import numpy as np
import os
import re
import commentjson
import glob
import cv2
from pathlib import Path
from PyQt5 import QtCore, QtGui
from PyQt5.QtCore import pyqtSlot
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *

from .Design import create_ui

from utils.general_util import load_calib_data, json_dump, json_load
from utils.triang import triangulate_robust


def _tryint(s):
    try:
        return int(s)
    except:
        return s


def _alphanum_key(s):
    """ Turn a string into a list of string and number chunks.
        "z23a" -> ["z", 23, "a"]
    """
    return [_tryint(c) for c in re.split('([0-9]+)', s)]


def sort_nicely(l):
    """ Sort the given list in the way that humans expect.
    """
    l.sort(key=_alphanum_key)


def shorten_path(file_path, max_length=40):
    numparts = len(Path(file_path).parts)
    # print('Path has %d parts' % numparts)

    for i in range(numparts, 1, -1):
        # print('Trying %d parts' % i)
        tmp = str(_shorten_path(file_path, i))
        # print('%s has %d length' % (tmp, len(tmp)))
        if len(tmp) <= max_length:
            return '../' + str(tmp)


def _shorten_path(file_path, length):
    return Path(*Path(file_path).parts[-length:])


class App(QWidget):
    def __init__(self, model, param, data_path, anno_file, calib_file):
        super().__init__()
        self.anno_file = os.path.join(data_path, anno_file)
        self.calib_file = os.path.join(data_path, calib_file)

        self.anno = dict()
        self.cam_ids = list()
        self.calib = dict()
        self.solution3d = dict()
        self.frames = None

        # check given arguments
        self.config = self.read_check_config(data_path, model, param)
        self.data = self.read_check_data(data_path)
        self.calib_lists = self.calib_to_lists(
            load_calib_data(self.calib_file, return_cam2world=False)
        )

        # creates all the buttons and stuff
        create_ui(self, self.config)

        # populate lists
        self.setup_kp_list()
        self.setup_file_list()

        # init state
        self.frame_id = 0
        self.set_file_list_selection(self.frame_id)
        self.update_visibility()
        self.read_frames()
        self.update_frame_view()
        self.update_progress_label()
        self.set_file_list_selection(self.frame_id)
        self.kp_list.selectAll()

        self.show()
        self.frame_view.fitInView() # call to have resized images right away

        # name buttons and connect them to their actions
        self.connect_buttons([
            ('btn_del', self.btn_del_clicked, QtGui.QKeySequence.Delete),
            ('btn_tri', self.btn_tri_clicked, QtCore.Qt.Key_T),
            ('btn_save', self.btn_save_clicked, QtCore.Qt.Key_S),
            ('btn_load', self.btn_load_clicked, QtCore.Qt.Key_L),
            ('btn_prev', self.btn_prev_clicked, QtGui.QKeySequence.MoveToPreviousChar),
            ('btn_next', self.btn_next_clicked, QtGui.QKeySequence.MoveToNextChar)
        ])

        # name checkboxes and connect them to their actions
        self.connect_checkboxes([
            ('ckbox_repr', self.ckbox_repr_clicked, self.ckbox_repr_keypress, QtCore.Qt.Key_R),
            ('ckbox_center', self.ckbox_center_clicked, self.ckbox_center_keypress, QtCore.Qt.Key_C)
        ])

        # output basepath
        self.base_path_label.setText(
            shorten_path(data_path)
        )

        # shortcut all/None vis
        QShortcut(QtCore.Qt.Key_V, self, self.toggle_vis)

        # shortcut groups
        self.setup_keypoint_groups()

    """ INPUT READER FUNCTIONS. """
    def read_check_config(self, base_path, model, param):
        config = dict()
        for k, v in param.viewer.items():
            config[k] = v
        config['groups'] = model.viewer['groups']
        config['example_path'] = model.viewer['example_path']

        config['keypoints'] = list()
        for id, (name, color) in enumerate(model.keypoints):
            config['keypoints'].append(
                {
                    'name': name,
                    'coord': model.viewer['example_keypoint_coords'][id],
                    'color': color
                }
            )

        assert os.path.exists(self.calib_file), 'Calibbration file not found: %s' % self.calib_file
        return config

    def read_check_data(self, data_path):
        img_list = list()
        cam_dirs = list()
        self.cam_ids = list()
        for cid in range(100):
            tmp = os.path.join(data_path, 'cam%d' % cid)
            if os.path.exists(tmp):
                self.cam_ids.append(cid)
                cam_dirs.append(tmp)
        msg = 'No camera directories found from basepath: %s' % data_path
        assert len(cam_dirs) > 0, msg
        sort_nicely(cam_dirs)

        num_frames = None
        for c in cam_dirs:
            search_path = c + '/*.%s' % self.config['img_extension']
            frames = glob.glob(search_path)
            msg = 'Could not locate a single frame searching: %s' % search_path
            assert len(frames) > 0, msg
            sort_nicely(frames)
            if num_frames is None:
                num_frames = len(frames)
            msg = 'Number of frames differs between camera folders in %s' % data_path
            assert len(frames) == num_frames, msg
            img_list.append(frames)

        # transpose outer and inner list
        img_list = list([list(i) for i in zip(*img_list)])
        return img_list

    """ UI SETUP FUNCTIONS. """
    def setup_kp_list(self):
        # populate with keypoints
        self.kp_list.clear()
        for data in self.config['keypoints']:
            self.kp_list.addItem(data['name'])

        # connect with handler
        self.kp_list.itemSelectionChanged.connect(self.handle_list_selection_change)

    def setup_file_list(self):
        # populate with keypoints
        self.file_list.clear()
        for file_name in [x[0] for x in self.data]:
            self.file_list.addItem(os.path.basename(file_name))

        # connect with handler
        self.file_list.itemSelectionChanged.connect(self.handle_file_list_changed)

    def setup_keypoint_groups(self):
        shortcuts = [
            ('0', QtCore.Qt.Key_0),
            ('1', QtCore.Qt.Key_1),
            ('2', QtCore.Qt.Key_2),
            ('3', QtCore.Qt.Key_3),
            ('4', QtCore.Qt.Key_4),
            ('5', QtCore.Qt.Key_5),
            ('6', QtCore.Qt.Key_6),
            ('7', QtCore.Qt.Key_7),
            ('8', QtCore.Qt.Key_8),
            ('9', QtCore.Qt.Key_9)
        ]

        for s, key in shortcuts:
            if s in self.config['groups'].keys():
                kp_list = list(self.config['groups'][s])

                # def shortcut_factory(a, b):
                #     # print('setting up', a, b)
                #     QShortcut(a, self, lambda: self.set_group_vis(b))
                #
                # shortcut_factory(key, kp_list)

                QShortcut(key, self, lambda a=tuple(kp_list): self.set_group_vis(a))

    def connect_checkboxes(self, job_list):
        for ckbox_name, target_fct_click, target_fct_key, binding in job_list:
            ckbox_obj = self.findChild(QCheckBox, ckbox_name)

            ckbox_obj.clicked.connect(
                # lambda checked, a=ckbox_obj: target_fct_click(a)
                target_fct_click
            )

            if binding is not None:
                QShortcut(
                    binding,
                    ckbox_obj,
                    # lambda a=ckbox_obj: target_fct_key(a)
                    target_fct_key
                )

    def connect_buttons(self, job_list):
        for btn_name, target_fct, binding in job_list:
            btn_obj = self.findChild(QPushButton, btn_name)
            if binding is not None:
                QShortcut(binding, btn_obj, target_fct)
            btn_obj.clicked.connect(target_fct)

    def update_frame_view(self):
        # select visible points
        selected_kps = [item.text() for item in self.kp_list.selectedItems()]

        # set new images
        self.frame_view.set_images(
            self.frames,
            selected_kps
        )

    def get_frame_name(self):
        return os.path.basename(self.data[self.frame_id][0])

    def update_anno(self):
        """ Read current annotation from frame and put it into out anno member. """
        this_frame_anno = self.read_anno_from_frame()
        if len(this_frame_anno) > 0:
            frame_name = self.get_frame_name()
            self.anno[frame_name] = this_frame_anno

    def calib_to_lists(self, calib):
        K_list, dist_list, M_list = list(), list(), list()
        for cid in self.cam_ids:
            cam_name = 'cam%d' % cid
            K_list.append(calib[cam_name]['K'])
            dist_list.append(calib[cam_name]['dist'])
            M_list.append(calib[cam_name]['M'])
        return K_list, dist_list, M_list

    def read_anno_from_frame(self):
        kp_uv, vis2d, _ = self.anno2mat(only_vis_points=False)
        points3d, vis3d, _ = self.triangulate(kp_uv, vis2d)

        output = defaultdict(dict)
        for i, cid in enumerate(self.cam_ids):
            if np.sum(vis2d[i]) > 0:
                output['cam%d' % cid] = {
                    'kp_uv': kp_uv[i],
                    'vis': vis2d[i]
                }
        output['xyz'] = points3d
        output['vis3d'] = vis3d

        return dict(output)

    def read_frames(self):
        self.frames = [cv2.imread(x)[:, :, ::-1].copy() for x in self.data[self.frame_id]]

    def set_frame_id_save(self, new_val):
        # set frame id while staying within the valid value range
        self.frame_id = max(0, min(new_val, len(self.data)-1))

    def set_file_list_selection(self, frame_id):
        self.file_list.clearSelection()
        self.file_list.item(frame_id).setSelected(True)
        self.file_list.scrollToItem(
            self.file_list.item(frame_id),
            QAbstractItemView.PositionAtTop
        )

    def block_all_signals(self, val):
        self.file_list.blockSignals(val)
        self.kp_list.blockSignals(val)

    def update_progress_label(self):
        self.progress_label.setText('%d / %d' % (self.frame_id+1, len(self.data))) #change progress string

    def update_frame(self, frame_id_new=None):
        """ Whenever we need to update the frame to another image to be shown. """
        self.block_all_signals(True)

        self.update_anno()  # save current annotation
        self.set_frame_id_save(frame_id_new) # set new frame id
        self.set_file_list_selection(self.frame_id) # update file selection list
        self.update_progress_label()  #change progress string
        self.frame_view.clear()  # clear current frame
        self.read_frames() # read new images
        self.update_frame_view() # show new images
        self.update_frame_view_by_anno()

        self.block_all_signals(False)

    def anno2mat(self, only_vis_points=True):
        kp_names = [x['name'] for x in self.config['keypoints']]
        num_cam = len(self.data[0])
        num_kp = len(self.config['keypoints'])

        kp_uv = np.zeros((num_cam, num_kp, 2))
        vis_uv = np.zeros((num_cam, num_kp))
        for ident, data in self.frame_view.frame_keypoints.items():
            kp_id = kp_names.index(data.kp_name)

            if data.is_valid:
                if data.isVisible() or not only_vis_points:
                    pos = data.scenePos()
                    (u, v), _ = self.frame_view.stitch_img.map_stitch2orig(pos.x(), pos.y())
                    kp_uv[data.cid, kp_id, 0] = u
                    kp_uv[data.cid, kp_id, 1] = v
                    vis_uv[data.cid, kp_id] = 1.0

        return kp_uv, vis_uv, kp_names

    def convert_to_dict(self, point3d, point3d_proj, vis3d, kp_names):
        point3d_dict = dict()
        point3d_proj_dict = defaultdict(dict)
        for kp_id, name in enumerate(kp_names):
            if vis3d[kp_id] > 0.5:
                point3d_dict[name] = point3d[kp_id, :]
                for cid in range(point3d_proj.shape[0]):
                    point3d_proj_dict[name][kp_id] = point3d_proj[:, kp_id, :]
        return point3d_dict, point3d_proj_dict

    def update_visibility(self):
        self.block_all_signals(True)

        selected_items = [item.text() for item in self.kp_list.selectedItems()]

        for ident, data in self.frame_view.frame_keypoints.items():
            if any([k == data.kp_name for k in selected_items]):
                data.setVisible(True)
            else:
                data.setVisible(False)

        for kp in self.example_view.example_keypoints:
            if any([k == kp.id for k in selected_items]):
                kp.setVisible(True)
            else:
                kp.setVisible(False)

        self.block_all_signals(False)

    def update_frame_view_by_anno(self):
        self.frame_view.clear()  # clear frame view

        this_anno = self.anno.get(self.get_frame_name(), dict())
        for k, anno_cam in this_anno.items():
            if 'cam' not in k:
                continue
            match = re.findall('cam([\d]+)', k)
            assert len(match) == 1, 'Should always hold.'
            cid = int(match[0])
            c = self.cam_ids.index(cid)

            kp_uv = np.array(anno_cam['kp_uv'])
            vis = np.array(anno_cam['vis'])

            for kp_id, (uv, v) in enumerate(zip(kp_uv, vis)):
                if v < 0.5:
                    continue
                kp = self.config['keypoints'][kp_id]
                self.frame_view.update_frame_keypoint(c, kp, uv, is_scene_pos=False)

    """ INPUT HANDLERS. """
    @pyqtSlot()
    def btn_del_clicked(self):
        self.frame_view.clear()

    def triangulate(self, kp_uv, vis2d):
        K_list, dist_list, M_list = self.calib_lists
        points3d, _, vis3d, points2d_merged, vis_merged = triangulate_robust(kp_uv, vis2d, K_list, M_list)
        return points3d, vis3d, points2d_merged

    @pyqtSlot()
    def btn_tri_clicked(self):
        # get annotated information
        kp_uv, vis2d, kp_names = self.anno2mat()

        # triangulate
        points3d, vis3d, points2d_merged = self.triangulate(kp_uv, vis2d)
        # print('Triangulation yields %d 3D points' % np.sum(vis3d))

        # pass results to image
        self.frame_view.points3d, self.frame_view.points3d_proj = self.convert_to_dict(points3d, points2d_merged, vis3d, kp_names)

        # enforce repr button to be checked and frame_view to draw
        item = self.findChild(QCheckBox, 'ckbox_repr')
        item.setChecked(True)
        self.frame_view.draw_repr = True

        # update frame view
        self.update_frame_view()

    @pyqtSlot()
    def btn_save_clicked(self):
        self.update_anno()
        do_save = True
        if os.path.exists(self.anno_file):
            reply = QMessageBox.question(
                self,
                'Save Annotations',
                'File does already exist. Do you want to overwrite: %s' % self.anno_file
            )
            if reply == QMessageBox.No:
                do_save = False

        if do_save:
            json_dump(self.anno_file, self.anno, verbose=True)

    @pyqtSlot()
    def btn_load_clicked(self):
        self.anno = json_load(self.anno_file)
        self.update_frame_view_by_anno()
        self.update_visibility()

    @pyqtSlot()
    def btn_prev_clicked(self):
        self.update_frame(self.frame_id - 1)

    @pyqtSlot()
    def btn_next_clicked(self):
        self.update_frame(self.frame_id + 1)

    @pyqtSlot()
    def ckbox_repr_clicked(self):
        item = self.findChild(QCheckBox, 'ckbox_repr')
        if item.isChecked():
            self.frame_view.draw_repr = True
        else:
            self.frame_view.draw_repr = False

        self.update_frame_view()

    def ckbox_repr_keypress(self):
        item = self.findChild(QCheckBox, 'ckbox_repr')
        item.blockSignals(True)
        if item.isChecked():
            self.frame_view.draw_repr = False
            item.setChecked(False)
        else:
            self.frame_view.draw_repr = True
            item.setChecked(True)
        item.blockSignals(False)

        self.update_frame_view()

    def set_draw_center(self, val):
        for items in self.frame_view.frame_keypoints.values():
            items.paint_center = val

    @pyqtSlot()
    def ckbox_center_clicked(self):
        item = self.findChild(QCheckBox, 'ckbox_center')
        if item.isChecked():
            self.set_draw_center(True)
        else:
            self.set_draw_center(False)

        self.update_frame_view()

    def ckbox_center_keypress(self):
        item = self.findChild(QCheckBox, 'ckbox_center')
        item.blockSignals(True)
        if item.isChecked():
            self.set_draw_center(False)
            item.setChecked(False)
        else:
            self.set_draw_center(True)
            item.setChecked(True)
        item.blockSignals(False)

        self.update_frame_view()

    @pyqtSlot()
    def handle_list_selection_change(self):
        self.update_visibility()

        # update frame view
        self.update_frame_view()

    def toggle_vis(self):
        if len(self.kp_list.selectedItems()) == self.kp_list.count():
            self.kp_list.clearSelection()
        else:
            self.kp_list.selectAll()

    def set_group_vis(self, kp_name_list):
        """ Set visibility as defined by the configs group. """
        # figure out if we want to set it true or false
        self.block_all_signals(True)

        set_to = True
        if all([self.kp_list.item(i).isSelected() for i in range(self.kp_list.count())]):
            set_to = False

        for i in range(self.kp_list.count()):
            if self.kp_list.item(i).text() in kp_name_list:
                self.kp_list.item(i).setSelected(set_to)
            else:
                self.kp_list.item(i).setSelected(False)
        self.block_all_signals(False)
        self.update_visibility()
        self.update_frame_view()

    def handle_file_list_changed(self):
        """ What happends when the file list selection changes:
            Set new frame id according to selection and then update the UI.
        """
        # figure out new frame id
        sel_id = 0
        for i in range(self.file_list.count()):
            if self.file_list.item(i).isSelected():
                sel_id = i
                break

        # do the ui update
        self.update_frame(sel_id)

