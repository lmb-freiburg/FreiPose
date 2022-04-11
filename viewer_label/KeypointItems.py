from PyQt5 import QtGui, QtCore, QtWidgets
from PyQt5.QtWidgets import *


def paint_circle(self, painter: QtGui.QPainter, option: 'QStyleOptionGraphicsItem', widget):
    pen = painter.pen()
    pen.setColor(self.color)
    pen.setWidth(self.width)
    painter.setPen(pen)
    painter.drawEllipse(0, 0, self.radius * 2, self.radius * 2)


def paint_circle_with_center(self, painter: QtGui.QPainter, option: 'QStyleOptionGraphicsItem', widget):
    pen = painter.pen()
    pen.setColor(self.color)
    pen.setWidth(self.width)
    painter.setPen(pen)
    painter.drawEllipse(0, 0, self.radius * 2, self.radius * 2)
    painter.drawLine(0, self.radius, 2 * self.radius, self.radius)
    painter.drawLine(self.radius, 0, self.radius, 2 * self.radius)


class KeypointItem(QtWidgets.QGraphicsWidget):
    def __init__(self, parent, id, radius, color, width):
        super(KeypointItem, self).__init__(parent)
        self.color = color
        self.radius = radius
        self.width = width
        self.id = id
        self.paint_center = False

        self.setAttribute(QtCore.Qt.WA_DeleteOnClose)

    def boundingRect(self):
        return QtCore.QRectF(-2*self.radius - self.width / 2, -2*self.radius - self.width / 2,
                             4*self.radius + self.width, 4*self.radius + self.width)

    def setPos(self, pos, y=None):
        if y is None:
            self.setX(pos.x())
            self.setY(pos.y())
        else:
            self.setX(pos)
            self.setY(y)

    def setX(self, x):
        super(KeypointItem, self).setX(x - self.radius)

    def setY(self, y):
        super(KeypointItem, self).setY(y - self.radius)

    def pos(self):
        return QtCore.QPointF(self.x(), self.y())

    def x(self):
        return super(KeypointItem, self).x() + self.radius

    def y(self):
        return super(KeypointItem, self).y() + self.radius

    def scenePos(self):
        scenePos = super(KeypointItem, self).scenePos()
        return QtCore.QPointF(scenePos.x() + self.radius, scenePos.y() + self.radius)

    def paint(self, *args):
        if self.paint_center:
            paint_circle_with_center(self, *args)
        else:
            paint_circle(self, *args)


class ExampleKeypointItem(KeypointItem):
    """ This creates a drag, once you click on it. """
    def __init__(self, parent, id, x, y, radius, color, width):
        super(ExampleKeypointItem, self).__init__(parent, id, radius, color, width)
        self.example_x = x
        self.example_y = y
        self.setPos(self.example_x, self.example_y)

    def render(self):
        rect = self.boundingRect().toRect()
        if rect.isNull() or not rect.isValid():
            return QtGui.QPixmap()

        # Create the pixmap
        pixmap = QtGui.QPixmap(rect.size())
        pixmap.fill(QtCore.Qt.transparent)

        # Render
        painter = QtGui.QPainter(pixmap)
        painter.setRenderHint(QtGui.QPainter.Antialiasing, True)
        painter.setRenderHint(QtGui.QPainter.TextAntialiasing, True)
        painter.translate(-rect.topLeft())
        self.paint(painter, QtWidgets.QStyleOptionGraphicsItem(), pixmap)
        painter.end()

        return pixmap

    def mousePressEvent(self, event):
        """ When a left click is made on a example keypoint we create a drag event, which can be captured by GraphicsView. """
        # print('Press event for generic ExampleKeypointItem: ', self.id)
        if event.button() == QtCore.Qt.LeftButton:
            drag = QtGui.QDrag(event.widget())
            mime_data = QtCore.QMimeData()
            mime_data.setText(self.id)
            drag.setMimeData(mime_data)
            drag.setPixmap(self.render())
            drag.setHotSpot(QtCore.QPoint(drag.pixmap().width()/2 + self.radius,
                            drag.pixmap().height()/2 + self.radius))
            drag.exec()
            event.accept()

        else:
            super(ExampleKeypointItem, self).mousePressEvent(event)


class FrameKeypointItem(KeypointItem):
    """ This is simply an movable object (no real drag and drop)."""
    kpReset = QtCore.pyqtSignal(str)
    kpMove = QtCore.pyqtSignal(str)

    def __init__(self, parent, id, radius, color, width, kp_name, cid):
        super(FrameKeypointItem, self).__init__(parent, id, radius, color, width)
        self.kp_name = kp_name
        self.cid = cid
        self.setFlag(QtWidgets.QGraphicsItem.ItemIsMovable, True)
        self.setVisible(False)
        self.is_valid = False

    def mouseReleaseEvent(self, event: 'QGraphicsSceneMouseEvent'):
        """ Handles moves besides the one at creation time. """
        if event.button() == QtCore.Qt.LeftButton:
            super(KeypointItem, self).mouseReleaseEvent(event)
            self.kpMove.emit(self.id)

    def mousePressEvent(self, event):
        """ Handles resets. """
        if event.button() == QtCore.Qt.RightButton:
            self.setVisible(False)
            self.is_valid = False
            self.kpReset.emit(self.id)


