from PyQt5 import QtGui, QtCore, QtWidgets
from PyQt5.QtWidgets import *
import re
import numpy as np
import cv2

from .KeypointItems import *
from utils.StitchedImage import StitchedImage


class FrameView(QGraphicsView):
    """ The graphics view intem used to show the image tb annotated. """
    def __init__(self, parent,
                 width, height,
                 keypoints, kp_radius, kp_width, verbose=False):
        super(FrameView, self).__init__(parent)
        # size how large the image is in the Qt frame
        self.width = width
        self.height = height

        self.keypoints = keypoints
        self.all_names = [x['name'] for x in self.keypoints]
        self.stitch_img = None
        self.draw_repr = True
        self.points3d = dict()
        self.points3d_proj = dict()
        self.frame_keypoints = dict()

        self.kp_radius = kp_radius
        self.kp_width = kp_width
        self.verbose = verbose

        self.setAcceptDrops(True)  # make this class accept drags

        self.reset_all_kp = False
        self._zoom_scale_factor = 1.0

        # create scene and fill with image
        main_scene = QGraphicsScene(self)
        main_scene.setBackgroundBrush(QtGui.QBrush(QtCore.Qt.black))

        self.photo = QtWidgets.QGraphicsPixmapItem()
        main_scene.addItem(self.photo)
        self.setSceneRect(0, 0, self.width, self.height)
        self.setScene(main_scene)
        self.main_scene = main_scene

        # settings
        self.setTransformationAnchor(QtWidgets.QGraphicsView.AnchorUnderMouse)
        self.setResizeAnchor(QtWidgets.QGraphicsView.AnchorUnderMouse)
        self.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.setBackgroundBrush(QtGui.QBrush(QtGui.QColor(30, 30, 30)))
        self.setFrameShape(QtWidgets.QFrame.NoFrame)
        self.setRenderHints(QtGui.QPainter.Antialiasing | QtGui.QPainter.SmoothPixmapTransform)

    def clear(self):
        self.points3d = dict()
        self.points3d_proj = dict()
        for kp_name in list(self.frame_keypoints.keys()):
            self.delete_frame_keypoint(kp_name)

    def update_frame_keypoint(self, cid, kp, pos, is_scene_pos=True):
        ident = '%d_%s' % (cid, kp['name'])
        if ident not in self.frame_keypoints:
            if self.verbose:
                print('Creating frame keypoint', ident)
            # create because not existant yet
            self.frame_keypoints[ident] = FrameKeypointItem(
                self.photo,
                ident,
                self.kp_radius,
                QtGui.QColor.fromRgb(*kp['color']),
                self.kp_width,
                kp['name'],
                cid
            )
            self.frame_keypoints[ident].kpMove.connect(self.handle_kp_moved)
            self.frame_keypoints[ident].kpReset.connect(self.handle_kp_reset)

        if not is_scene_pos:
            # usually we get a scenepos except for when it was created by the app, then we have to do conversion
            # map from original image to stitch
            u, v = self.stitch_img.map_orig2stitch(pos[0], pos[1], cid)
            pos = QtCore.QPoint(u, v)

        # move to location
        self.frame_keypoints[ident].setPos(pos)
        self.frame_keypoints[ident].setVisible(True)
        self.frame_keypoints[ident].is_valid = True

    def handle_kp_moved(self, kp_name):
        pos = self.frame_keypoints[kp_name].scenePos()
        cid = self.stitch_img._get_subframe_id(pos.x(), pos.y())
        if self.verbose:
            print(kp_name, 'moved to subframe ', cid, ' with scene location ', pos.x(), pos.y())

        if cid == -1:
            if self.verbose:
                print('Special case: moved to an illegal area -> delete')
            # point moved to an invalid area
            # self.frame_keypoints.pop(kp_name)
            self.delete_frame_keypoint(kp_name)

        else:
            # check if it moved into another subframe
            cid_old, kp_name_stripped = self.frame_keypoints[kp_name].cid, self.frame_keypoints[kp_name].kp_name

            if cid_old != cid:
                if self.verbose:
                    print('Special case: moved into a new area')
                kp_name_new = '%d_%s' % (cid, kp_name_stripped)
                if kp_name_new in self.frame_keypoints.keys():
                    # is this subframe was already labeled, remove the old entry
                    self.delete_frame_keypoint(kp_name_new)

                kp = self.frame_keypoints[kp_name]
                self.frame_keypoints[kp_name_new] = FrameKeypointItem(
                                                        self.photo,
                                                        kp_name_new,
                                                        self.kp_radius,
                                                        kp.color,
                                                        self.kp_width,
                                                        kp.kp_name,
                                                        cid
                                                    )
                self.frame_keypoints[kp_name_new].kpMove.connect(self.handle_kp_moved)
                self.frame_keypoints[kp_name_new].kpReset.connect(self.handle_kp_reset)

                # move to location
                self.frame_keypoints[kp_name_new].setPos(pos)
                self.frame_keypoints[kp_name_new].setVisible(True)
                self.frame_keypoints[kp_name_new].is_valid = True

                self.delete_frame_keypoint(kp_name)

    def handle_kp_reset(self, kp_name):
        modifiers = QApplication.keyboardModifiers()
        if self.reset_all_kp or modifiers == QtCore.Qt.ShiftModifier:
            # remove all keypoints of a certain type across all views
            remove_list = list()
            for k, frame_kp in self.frame_keypoints.items():
                if frame_kp.kp_name in kp_name:
                    remove_list.append(k)

            for k in remove_list:
                self.delete_frame_keypoint(k)

        else:
            self.delete_frame_keypoint(kp_name)

    def delete_frame_keypoint(self, kp_name):
        self.frame_keypoints[kp_name].close()  # this deletes the GraphicWidget
        self.frame_keypoints.pop(kp_name, None)  # remove it from our dict

        # remove it as child of the Pixmap item
        dummyParent = QtWidgets.QGraphicsPixmapItem()
        for item in self.photo.childItems():
            if type(item) == FrameKeypointItem:
                if item.id == kp_name:
                    item.setParentItem(dummyParent)  # sets a new parent item which allows for deletion of this item
        del dummyParent

    def set_images(self, img_list, selected_kps):
        self.stitch_img = StitchedImage(img_list, target_size='max')
        img_np = self.stitch_img.image

        # draw reprojections
        if self.draw_repr:
            img_np = self.draw_reprojections(img_np, selected_kps)

        # show stitch
        image = QtGui.QImage(img_np.data, img_np.shape[1], img_np.shape[0], 3 * img_np.shape[1],
                             QtGui.QImage.Format_RGB888)
        image = image.scaled(img_np.shape[1], img_np.shape[0], QtCore.Qt.IgnoreAspectRatio)
        self.photo.setPixmap(QtGui.QPixmap.fromImage(image))

    def mousePressEvent(self, event: QtGui.QMouseEvent):
        """ For panning the main view. """
        if event.button() == QtCore.Qt.LeftButton:
            self.setDragMode(QtWidgets.QGraphicsView.ScrollHandDrag)
            event.accept()
        super(FrameView, self).mousePressEvent(event)

    def mouseReleaseEvent(self, event: QtGui.QMouseEvent):
        """ For stop panning the main view. """
        if event.button() == QtCore.Qt.LeftButton:
            self.setDragMode(QtWidgets.QGraphicsView.NoDrag)
            event.accept()
        super(FrameView, self).mouseReleaseEvent(event)

    def dragEnterEvent(self, event):
        """ For when an example keypoint is dragged into the view. """
        if event.mimeData().hasFormat('text/plain'):
            event.accept()
            # print('Drag enter', event.mimeData().text())
        else:
            event.ignore()
            # print('Drag enter no data')

    def dropEvent(self, event):
        """ For when an example keypoint is dropped into the view. """
        # print('FrameView: dropEvent')
        ex_kp_name = str(event.mimeData().text())
        assert ex_kp_name in self.all_names, 'Name should be defined.'
        pos = self.mapToScene(event.pos())
        cid = self.stitch_img._get_subframe_id(pos.x(), pos.y())
        idx = self.all_names.index(ex_kp_name)
        self.update_frame_keypoint(cid, self.keypoints[idx], pos)
        event.accept()
        super(FrameView, self).dropEvent(event)

    def dragMoveEvent(self, event: QtGui.QDragMoveEvent):
        # apparently one has to have this function overwise the dropevent does not work.
        pass

    def wheelEvent(self, event):
        """ Zoom functionality."""
        if not self.photo.pixmap().isNull():
            # direction of scrolling
            if event.angleDelta().y() > 0:
                # zooming in
                delta_zoom = 1.25
            else:
                # zooming out
                delta_zoom = 0.8

            # change zoom
            self.updateZoomedFrame(delta_zoom)

    def updateZoomedFrame(self, delta_zoom=1.0, reset=False):
        """ Scales the frame view according to zoom_scale_factor. """
        # reset?
        if reset:
            self._zoom_scale_factor = 1.0
            self.fitInView()
        else:
            # keep track of overall zoom
            self._zoom_scale_factor *= delta_zoom

            # we cant make it bigger than the reference size
            if self._zoom_scale_factor < 1.0:
                self._zoom_scale_factor = 1.0
                self.fitInView()
            else:
                # scale to target scale and cut the part we want
                self.scale(delta_zoom, delta_zoom)
                self.main_scene.setSceneRect(QtCore.QRectF())

    def fitInView(self, *args):
        rect = QtCore.QRectF(self.photo.pixmap().rect())
        if not rect.isNull():
            self.setSceneRect(QtCore.QRectF(QtCore.QPoint(0, 0), rect.size()))
            unity = self.transform().mapRect(QtCore.QRectF(0, 0, 1, 1))
            self.scale(1 / unity.width(), 1 / unity.height())
            viewrect = self.viewport().rect()
            scenerect = self.transform().mapRect(rect)
            factor = min(viewrect.width() / scenerect.width(),
                         viewrect.height() / scenerect.height())
            self.scale(factor, factor)
            self.centerOn(rect.center())

    def draw_reprojections(self, img_np, selected_kps):
        thickness = int(round(self.kp_width))
        r = int(round(self.kp_radius))

        for name, pts2d in self.points3d_proj.items():
            if name not in selected_kps:
                continue

            kp_id = self.all_names.index(name)

            for cid, pt2d in enumerate(pts2d[kp_id]):
                u, v, in_bounds = self.stitch_img.map_orig2stitch(pt2d[0], pt2d[1], cid, return_bounds=True)

                if in_bounds:
                    u, v = u.round().astype(np.int32), v.round().astype(np.int32)
                    c = self.keypoints[kp_id]['color']
                    img_np = cv2.line(img_np, (u-r, v-r), (u+r, v+r), color=c, thickness=thickness)
                    img_np = cv2.line(img_np, (u+r, v-r), (u-r, v+r), color=c, thickness=thickness)
        return img_np
