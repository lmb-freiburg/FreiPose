import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
from pathlib import Path
from PyQt5 import QtCore, QtGui
from PyQt5.QtCore import pyqtSlot
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from joblib import Parallel, delayed
import time

from .Design import create_ui

from utils.general_util import sample_uniform, my_mkdir, json_dump, json_load
import utils.CamLib as cl
from utils.triang import triangulate_robust
from utils.Precacher import Precacher


def read_vid_length(video_path):
    """ Reads a single frame from a video.
    """
    cap = cv2.VideoCapture(video_path)
    vid_size = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    return vid_size


def read_vid_frame(video_path, fid):
    """ Reads a single frame from a video.
    """
    cap = cv2.VideoCapture(video_path)
    vid_size = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    assert 0 <= fid < vid_size, 'Frame id is outside the video.'

    cap.set(cv2.CAP_PROP_POS_FRAMES, fid)
    for i in range(5):
        suc, img = cap.read()
        if not suc:
            print('Reading video frame was not successfull. Will try again in 2 sec.')
            time.sleep(2)
        else:
            break
    assert img is not None and suc, 'Reading not successful'
    cap.release()
    return img


def read_video_sample(vid_files, fid, cam_range, calib_file, read_dist=True, read_parallel=False):
    # read calib
    calib_path = os.path.join(os.path.dirname(vid_files[0]), calib_file)
    calib = json_load(calib_path)

    K_list = [np.array(calib['K']['cam%d' % cid]) for cid in cam_range]
    M_list = [np.linalg.inv(np.array(calib['M']['cam%d' % cid])) for cid in cam_range]
    if read_dist:
        dist_list = [np.array(calib['dist']['cam%d' % cid]) for cid in cam_range]

    # read image
    img_list = list()
    if read_parallel:
        img_list = Parallel(n_jobs=len(cam_range))(delayed(read_vid_frame)(vid, fid) for vid in vid_files)

    else:
        for vid in vid_files:
            img_list.append(
                read_vid_frame(vid, fid)
            )

    if read_dist:
        return img_list, K_list, M_list, dist_list
    return img_list, K_list, M_list


def shorten_path(file_path, num_parts=1):
    numparts = len(Path(file_path).parts)
    if numparts > num_parts:
        tmp = str(_shorten_path(file_path, num_parts))
        return '../' + str(tmp)
    return file_path


def _shorten_path(file_path, length):
    return Path(*Path(file_path).parts[-length:])


class App(QWidget):
    def find_valid_pred(self, predictions):
        valid_keys = list()
        for k, v in enumerate(predictions):
            if 'kp_xyz' in v.keys() and 'kp_score' in v.keys():
                valid_keys.append(k)

        return valid_keys

    def __init__(self, pred_file, model, param,
                 predictions, video_list, cam_range, calib_file):
        super().__init__()

        # check given arguments
        self.config = self.read_check_config(model, param)

        self.output_task_dir = 'labeled_set%d'

        # case there are no predictions yet
        if predictions is None:
            # empty = {''}
            predictions = [{} for _  in range(read_vid_length(video_list[0]))]

        # data containers
        self.pred_file = pred_file
        self.key_id = 0
        self.cam_range = cam_range
        self.calib_file = calib_file
        self.video_list = video_list
        self.video_list_short = [shorten_path(v) for v in video_list]
        self.predictions = predictions
        self.keys = [i for i in range(len(predictions))]

        self.precacher = Precacher(
            self.keys,
            lambda x: read_video_sample(self.video_list, x, self.cam_range, self.calib_file)
        )
        self.precacher.start()

        # find unlabeled instances
        self.keys_valid = self.find_valid_pred(self.predictions)
        print('%d out of %d frames contains valid predictions (%d missing)' % (len(self.keys_valid),
                                                                               len(self.keys),
                                                                               len(self.keys) - len(self.keys_valid)))

        # data item to create output
        self.label_tasks = dict()  # {'vid_key': {'xyz': X, 'vis': Y} }

        self.K_list, self.M_list, self.dist_list = None, None, None

        # creates all the buttons and stuff
        create_ui(self, self.config)
        self.frame_view.reset_all_kp = True # reset all keypoints jointly

        # populate lists
        self.setup_kp_list()
        self.setup_all_frames_list()

        # init state
        self.read_frames()  # read images
        self.set_file_list_selection(self.key_id) # set the file we read
        self.update_visibility()
        self.update_frame_view()
        self.update_frame_view_points()
        self.update_file_list_selection()
        self.update_histogram()
        self.file_list_sel_full_keys = list()

        # progress bar
        self.pb_length = None
        self.pb_cnt = 0

        self.show()
        self.frame_view.fitInView() # call to have resized images right away

        # name buttons and connect them to their actions
        self.connect_buttons([
            ('btn_del_frame', self.btn_delete_all, QtGui.QKeySequence.Delete),
            ('btn_rec_frame', lambda: self.update_frame_view_points(use_labels=False), QtCore.Qt.Key_R),
            ('btn_sel_uni', self.btn_sel_uni, None),
            ('btn_sel_score', self.bn_sel_score, None),
            ('bn_sel_curr', self.bn_sel_curr, None),
            ('bn_unselect', self.bn_unselect, QtCore.Qt.Key_U),
            ('bn_unselect_all', self.bn_unselect_all, QtCore.Qt.Key_U),
            ('btn_sort_id', self.btn_sort_id, QtCore.Qt.Key_N),
            ('btn_sort_score', self.btn_sort_conf, QtCore.Qt.Key_M),
            ('btn_write', self.btn_write, None)
        ])

        # shortcut all/None vis
        QShortcut(QtCore.Qt.Key_V, self, self.toggle_vis)

        # shortcut next/prev
        QShortcut(QtGui.QKeySequence.MoveToNextChar, self, self.btn_next_clicked)
        QShortcut(QtGui.QKeySequence.MoveToPreviousChar, self, self.btn_prev_clicked)

        # shortcuts
        QShortcut(QtCore.Qt.Key_S, self, self.bn_sel_curr)
        QShortcut(QtCore.Qt.Key_W, self, self.btn_write)

        # shortcut groups
        self.setup_keypoint_groups()

        # connect selected frames
        self.file_list_sel.itemSelectionChanged.connect(self.handle_file_list_selection_change)

    """ INPUT READER FUNCTIONS. """

    def read_check_config(self, model, param):
        config = dict()
        for k, v in param.viewer.items():
            config[k] = v
        config['groups'] = model.viewer['groups']

        config['keypoints'] = list()
        for id, (name, color) in enumerate(model.keypoints):
            config['keypoints'].append(
                {
                    'name': name,
                    'coord': model.viewer['example_keypoint_coords'][id],
                    'color': color
                }
            )

        return config

    """ UI SETUP FUNCTIONS. """
    def setup_kp_list(self):
        # populate with keypoints
        self.kp_list.clear()
        for data in self.config['keypoints']:
            self.kp_list.addItem(data['name'])
        self.kp_list.selectAll()

        # connect with handler
        self.kp_list.itemSelectionChanged.connect(self.handle_kp_list_selection_change)

    def setup_all_frames_list(self):
        self.block_all_signals(True)

        # populate with keypoints
        self.file_list.clear()
        for k in self.keys:
            self.file_list.addItem(str(k))
            if k not in self.keys_valid:
                self.file_list.item(self.file_list.count()-1).setForeground(QtCore.Qt.red)

        # connect with handler
        self.file_list.itemSelectionChanged.connect(self.handle_file_list_change)
        self.block_all_signals(False)

    def connect_buttons(self, job_list):
        for btn_name, target_fct, binding in job_list:
            btn_obj = self.findChild(QPushButton, btn_name)
            if binding is not None:
                QShortcut(binding, btn_obj, target_fct)
            btn_obj.clicked.connect(target_fct)

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

                QShortcut(key, self, lambda a=tuple(kp_list): self.handle_group_vis_change(a))

    """ UI HANDLER FUNCTIONS. """
    @pyqtSlot()
    def btn_placeholder(self):
        print('placeholder')

    @pyqtSlot()
    def btn_sel_uni(self):
        edit_sample = self.findChild(QLineEdit, 'edit_sample')
        num_samples = int(edit_sample.text())

        # repopulate the selected file list
        self.file_list_sel.clear()
        for k in sample_uniform(self.keys, min(num_samples, len(self.keys))):
            self.file_list_sel_full_keys.append(k)

        self.populate_file_list_sel()

    @staticmethod
    def _key_dist(key, all_keys):
        min_dist = float('inf')
        for k in all_keys:
            d = abs(k - key)
            min_dist = min(min_dist, d)

        return min_dist

    @pyqtSlot()
    def bn_sel_score(self):
        edit_sample = self.findChild(QLineEdit, 'edit_sample')
        num_samples = int(edit_sample.text())

        edit_min_dist = self.findChild(QLineEdit, 'edit_min_dist')
        min_dist = int(edit_min_dist.text())

        conf = [np.mean(self.predictions[k]['kp_score']) for k in self.keys_valid]
        keys_sorted, _ = zip(*sorted(zip(self.keys_valid, conf), key=lambda x: x[1]))

        keys_invalid = [k for k in self.keys if k not in self.keys_valid]  # get invalid ones

        # accumulate invalid first and then according to score sort
        keys_all = list(keys_invalid)
        keys_all.extend(keys_sorted)

        # repopulate the selected file list
        self.file_list_sel.clear()
        items_picked = list()
        for k in keys_all:
            if self._key_dist(k, items_picked) < min_dist:
                continue

            self.file_list_sel_full_keys.append(k)
            items_picked.append(k)

            if len(items_picked) == num_samples:
                break
        self.populate_file_list_sel()

    @pyqtSlot()
    def bn_sel_curr(self):
        selected_ids = [self.file_list.row(item) for item in self.file_list.selectedItems()]
        if len(selected_ids) > 0:
            selected_items_full_keys = [self.keys[i] for i in selected_ids]

            for item in selected_items_full_keys:
                self.file_list_sel_full_keys.append(item)

            self.populate_file_list_sel()


    @pyqtSlot()
    def bn_unselect(self):
        selected_items = [item.text() for item in self.file_list_sel.selectedItems()]

        remove_items = list()
        for i in range(self.file_list_sel.count()):
            if self.file_list_sel.item(i).text() in selected_items:
                remove_items.append(i)

        for i in remove_items[::-1]:
            self.file_list_sel_full_keys.pop(i)
            self.file_list_sel.takeItem(i)


    @pyqtSlot()
    def bn_unselect_all(self):
        for _ in range(self.file_list_sel.count()):
            self.file_list_sel_full_keys.pop(0)
            self.file_list_sel.takeItem(0)

    @pyqtSlot()
    def btn_delete_all(self):
        self.frame_view.clear()  # clear frame view

    @pyqtSlot()
    def btn_sort_id(self):
        self.block_all_signals(True)
        self.save_label_state()

        # change sorting of keys
        self.sort_keys_by_id()

        # repopulate the file list
        self.file_list.clear()
        for k in self.keys:
            self.file_list.addItem(str(k))
        self.key_id = 0
        self.set_file_list_selection(self.key_id) # set the file we read

        self.block_all_signals(False)
        self.update_frame(save_current=False)

    @pyqtSlot()
    def btn_sort_conf(self):
        self.block_all_signals(True)
        self.save_label_state()

        # change sorting of keys
        self.sort_keys_by_conf()

        # repopulate the file list
        self.file_list.clear()
        for k in self.keys:
            self.file_list.addItem(str(k))
        self.key_id = 0
        self.set_file_list_selection(self.key_id) # set the file we read

        self.block_all_signals(False)
        self.update_frame(save_current=False)

    @pyqtSlot()
    def btn_prev_clicked(self):
        self.update_frame(self.key_id - 1)

    @pyqtSlot()
    def btn_next_clicked(self):
        self.update_frame(self.key_id + 1)

    def pb_start(self, length):
        # self.pb_label.setVisible(True)
        # self.pb.setVisible(True)
        self.pb_length = length
        self.pb_cnt = 0
        self.pb.setValue(0)
        qApp.processEvents(QtCore.QEventLoop.AllEvents, 50)
        # QApplication.processEvents()
        # self.processEvents()

    def pb_update(self):
        self.pb_cnt += 1
        self.pb.setValue(int(self.pb_cnt/self.pb_length*100))
        qApp.processEvents(QtCore.QEventLoop.AllEvents, 50)
        # QApplication.processEvents()
        # self.processEvents()

    def pb_finish(self):
        # self.pb_label.setVisible(False)
        # self.pb.setVisible(False)
        self.pb.setValue(100)

    @pyqtSlot()
    def btn_write(self):
        self.save_label_state()  # save current annotation

        num_kp = len(self.config['keypoints'])
        empty = {
            'kp_xyz': np.zeros((num_kp, 3)),
            'vis3d': np.zeros((num_kp, ))
        }

        # assemble all info we want to write to disk
        output_data = dict()
        for k in self.file_list_sel_full_keys:
            fid = int(k)
            if k in self.label_tasks.keys():
                output_data[fid] = self.label_tasks[k]

                # project into views
                for i, cid in enumerate(self.cam_range):
                    # project into frame
                    xyz = self.label_tasks[k]['kp_xyz']
                    kp_uv = cl.project(cl.trafo_coords(xyz, self.M_list[i]), self.K_list[i], self.dist_list[i])
                    output_data[fid]['cam%d' % cid] = {
                        'kp_uv': kp_uv,
                        'vis': self.label_tasks[k]['vis3d']
                    }

            else:
                output_data[fid] = empty

        self.pb_start(len(output_data))

        # figure out base path
        i = 0
        while True:
            base_path = os.path.join(
                os.path.dirname(self.video_list[0]),
                self.output_task_dir % i
            )
            if not os.path.exists(base_path):
                break
            i += 1

        # dump frames
        for fid, _ in output_data.items():
            img_list, K_list, M_list, dist_list = self.precacher.get_data(fid)

            # write image frames
            for cid, img in zip(self.cam_range, img_list):
                output_path = os.path.join(
                    base_path,
                    'cam%d' % cid,
                    '%08d.png' % fid
                )
                my_mkdir(output_path, is_file=True)
                cv2.imwrite(output_path, img)
                # print('Dumped: ', output_path)
            self.pb_update()

        self.pb_finish()

        # dump anno
        anno_out_path = os.path.join(
            base_path,
            'anno.json'
        )
        my_mkdir(anno_out_path, is_file=True)
        json_dump(anno_out_path,
                  {'%08d.png' % k: v for k, v in output_data.items()},
                  verbose=True)


    @pyqtSlot()
    def handle_kp_list_selection_change(self):
        self.update_visibility()

        # update frame view
        self.update_frame_view()

    @pyqtSlot()
    def handle_file_list_change(self):
        """ What happends when the file list selection changes:
            Set new frame id according to selection and then update the UI.
        """
        # figure out new frame id
        sel_id = None
        for i in range(self.file_list.count()):
            if self.file_list.item(i).isSelected():
                sel_id = i
                break

        # do the ui update
        self.update_frame(sel_id)

    @pyqtSlot()
    def handle_file_list_selection_change(self):
        """ What happends when the selected file list selection changes:
            Show the respective frame.
        """
        selected_items = [item.text() for item in self.file_list_sel.selectedItems()]

        assert len(selected_items) <= 1, 'This should always hold.'

        if len(selected_items) == 0:
            return

        selected_item = selected_items[0]

        # figure out frame id that was selected
        sel_id = -1
        for i in range(self.file_list.count()):
            if selected_item == self.file_list.item(i).text():
                sel_id = i
                break
        assert sel_id >= 0, 'Invalid selection id.'

        # do the ui update
        self.update_frame(sel_id)

    """ GENERAL FUNCTIONS """
    def sort_keys_by_id(self):
        # change sorting of keys
        self.keys = [i for i in range(len(self.predictions))]
        self.precacher.update_keylist(self.keys)

    def sort_keys_by_conf(self):
        # change sorting of keys
        keys = [i for i in range(len(self.predictions))]
        self.keys = [k for k in keys if k not in self.keys_valid]  # get invalid ones first

        # sort valid ones
        conf = [np.mean(self.predictions[k]['kp_score']) for k in self.keys_valid]
        tmp, _ = zip(*sorted(zip(self.keys_valid, conf), key= lambda x: x[1]))
        self.keys.extend(tmp)
        self.precacher.update_keylist(self.keys)

    def triangulate(self, kp_uv, vis2d):
        points3d, _, vis3d, points2d_merged, vis_merged = triangulate_robust(kp_uv, vis2d, self.K_list, self.M_list,
                                                                             dist_list=self.dist_list)
        return points3d, vis3d, points2d_merged

    def anno2mat(self, only_vis_points=True):
        kp_names = [x['name'] for x in self.config['keypoints']]
        num_cam = len(self.cam_range)
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

    def block_all_signals(self, val):
        self.file_list.blockSignals(val)
        self.file_list_sel.blockSignals(val)
        self.kp_list.blockSignals(val)

    def set_file_list_selection(self, key_id):
        self.block_all_signals(True)

        self.file_list.clearSelection()
        self.file_list.item(key_id).setSelected(True)
        self.file_list.scrollToItem(
            self.file_list.item(key_id),
            QAbstractItemView.PositionAtTop
        )
        self.block_all_signals(False)

    def read_frames(self):
        # img_list, self.K_list, self.M_list, self.dist_list = read_video_sample(self.video_list, self.keys[self.key_id],
        #                                                                        self.cam_range,
        #                                                                        read_dist=True, read_parallel=True)
        img_list, self.K_list, self.M_list, self.dist_list = self.precacher.get_data(self.keys[self.key_id])
        self.frames = [x[:, :, ::-1] for x in img_list]

    def update_frame_view(self):
        # select visible points
        selected_kps = [item.text() for item in self.kp_list.selectedItems()]

        # set new images
        self.frame_view.set_images(
            self.frames,
            selected_kps
        )

    def update_frame_view_points(self, use_labels=True):
        self.frame_view.clear()  # clear frame view
        k = self.keys[self.key_id]

        if k in self.keys_valid:
            # draw when there is a valid annotation

            this_pred = self.predictions[k]
            xyz = np.array(this_pred['kp_xyz'])[0]

            if k in self.label_tasks.keys() and use_labels:
                # use label task results if there are any
                this_pred = self.label_tasks[k]
                xyz = np.array(this_pred['kp_xyz'])

            for i, cid in enumerate(self.cam_range):
                # project into frame
                kp_uv = cl.project(cl.trafo_coords(xyz, self.M_list[i]), self.K_list[i], self.dist_list[i])

                for kp_id, uv in enumerate(kp_uv):
                    kp_name = self.config['keypoints'][kp_id]
                    self.frame_view.update_frame_keypoint(i, kp_name, uv, is_scene_pos=False)

            # make point all not movable
            for item in self.frame_view.frame_keypoints.values():
                item.setFlag(QGraphicsItem.ItemIsMovable, False)

    def update_visibility(self):
        self.block_all_signals(True)

        selected_items = [item.text() for item in self.kp_list.selectedItems()]

        for ident, data in self.frame_view.frame_keypoints.items():
            if any([k == data.kp_name for k in selected_items]):
                data.setVisible(True)
            else:
                data.setVisible(False)

        self.block_all_signals(False)

    def toggle_vis(self):
        if len(self.kp_list.selectedItems()) == self.kp_list.count():
            self.kp_list.clearSelection()
        else:
            self.kp_list.selectAll()

    def handle_group_vis_change(self, kp_name_list):
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

    def update_file_list_selection(self):
        self.block_all_signals(True)

        self.file_list.clearSelection()
        self.file_list.item(self.key_id).setSelected(True)
        self.file_list.scrollToItem(
            self.file_list.item(self.key_id),
            QAbstractItemView.PositionAtTop
        )

        self.block_all_signals(False)

    @staticmethod
    def fig2data(fig):
        """
        @brief Convert a Matplotlib figure to a 4D numpy array with RGBA channels and return it
        @param fig a matplotlib figure
        @return a numpy 3D array of RGBA values
        """
        # draw the renderer
        fig.canvas.draw ( )

        # Get the RGBA buffer from the figure
        w,h = fig.canvas.get_width_height()
        buf = np.fromstring ( fig.canvas.tostring_argb(), dtype=np.uint8 )
        buf.shape = ( w, h,4 )

        # canvas.tostring_argb give pixmap in ARGB mode. Roll the ALPHA channel to have it in RGBA mode
        buf = np.roll ( buf, 3, axis = 2 )
        return buf[:, :, :3]

    def update_histogram(self):
        conf = [np.mean(self.predictions[k]['kp_score']) for k in self.keys_valid]

        # calculate histogram
        count, edges = np.histogram(conf, bins=20)
        count = count / float(count.sum())
        values = 0.5 * (edges[:-1] + edges[1:])

        # draw histogram as matplotlib figure
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.stem(values, count*100)
        if self.keys[self.key_id] in self.keys_valid:
            i = self.keys_valid.index(self.keys[self.key_id])
            this_conf = conf[i]
            ax.plot([this_conf, this_conf],
                    [count.max()*100, 0.0],
                    'r')
        ax.set_xlabel('score in [0, 1]')
        ax.set_ylabel('occurence in %')
        figdata = self.fig2data(fig)
        plt.close(fig)

        figdata = np.copy(figdata)
        #figdata = cv2.resize(figdata, (300, 400), interpolation=cv2.INTER_LINEAR)
        height, width, channel = figdata.shape
        bytesPerLine = 3 * height
        image = QtGui.QImage(figdata.data, height, width, bytesPerLine, QImage.Format_RGB888).scaled(400, 300)
        pix = QtGui.QPixmap.fromImage(image)
        self.hist.setPixmap(pix)

    def read_anno_from_frame(self):
        kp_uv, vis2d, _ = self.anno2mat(only_vis_points=False)
        points3d, vis3d, _ = self.triangulate(kp_uv, vis2d)

        output = dict()
        output['kp_xyz'] = points3d
        output['vis3d'] = vis3d

        return dict(output)

    def save_label_state(self):
        """ Read current annotation from frame and put it into out anno member. """
        this_frame_name = self.keys[self.key_id]
        this_frame_anno = self.read_anno_from_frame()

        if len(this_frame_anno) > 0:
            self.label_tasks[this_frame_name] = this_frame_anno

    def set_key_id_save(self, new_val):
        if new_val is not None:
            # set frame id while staying within the valid value range
            self.key_id = max(0, min(new_val, len(self.keys)-1))
        return self.key_id

    def update_frame(self, key_id_new=None, save_current=True):
        """ Whenever we need to update the frame to another image to be shown. """
        self.block_all_signals(True)

        if save_current:
            self.save_label_state()  # save current annotation
        self.key_id = self.set_key_id_save(key_id_new) # set new key id
        self.set_file_list_selection(self.key_id) # update file selection list
        self.frame_view.clear()  # clear current frame
        self.read_frames() # read new images
        self.update_frame_view() # show new images
        self.update_frame_view_points()
        self.update_histogram()

        self.block_all_signals(False)

    def populate_file_list_sel(self):
        # make unique
        self.file_list_sel_full_keys = list(set(self.file_list_sel_full_keys))
        self.file_list_sel_full_keys = sorted(self.file_list_sel_full_keys)

        self.block_all_signals(True)
        self.file_list_sel.clear()
        for k in self.file_list_sel_full_keys:
            self.file_list_sel.addItem(str(k))
        self.block_all_signals(False)


