from PyQt5 import QtGui, QtCore, QtWidgets
from PyQt5.QtWidgets import *

from .KeypointItems import *


class ExampleView(QGraphicsView):
    """ The graphics view intem used to show the image tb annotated. """
    def __init__(self, parent, width, height, keypoints, kp_width, kp_radius):
        super(ExampleView, self).__init__(parent)
        self.width = width
        self.height = height
        self.keypoints = keypoints
        self.kp_width = kp_width
        self.kp_radius = kp_radius
        self.example_keypoints = list()

        self.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.setAlignment(QtCore.Qt.AlignTop | QtCore.Qt.AlignLeft)
    
        example_scene = QGraphicsScene(self)
        example_scene.setBackgroundBrush(QtGui.QBrush(QtCore.Qt.black))
        self.photo = QtWidgets.QGraphicsPixmapItem()
        example_scene.addItem(self.photo)
        self.setMinimumSize(width, height)
        self.setSceneRect(0, 0, self.width, self.height)
        self.setScene(example_scene)
        self.setRenderHints(QtGui.QPainter.Antialiasing | QtGui.QPainter.SmoothPixmapTransform)

    def set_image(self, img_np):
        image = QtGui.QImage(img_np.data, img_np.shape[1], img_np.shape[0], 3 * img_np.shape[1],
                             QtGui.QImage.Format_RGB888)
        image = image.scaled(self.width, self.height, QtCore.Qt.IgnoreAspectRatio)
        self.photo.setPixmap(QtGui.QPixmap.fromImage(image))

        s1 = float(self.width)/img_np.shape[1]
        s2 = float(self.height)/img_np.shape[0]
        
        for kp in self.keypoints:
            self.example_keypoints.append(
                ExampleKeypointItem(self.photo, kp['name'],
                                    s1*kp['coord'][0], s2*kp['coord'][1],
                                    self.kp_width,
                                    QtGui.QColor.fromRgb(*kp['color']),
                                    self.kp_radius)
            )