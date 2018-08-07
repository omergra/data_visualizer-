import cv2
from PyQt5 import QtCore
from PyQt5 import QtGui


class videoWidget(QtGui.QWidget):
    newFrame = QtCore.pyqtSignal(cv2.iplimage)

    def __init__(self, visualizer, parent=None):
        super(videoWidget, self).__init__(parent)
        self._frame = None
        self._visualizer = visualizer
        self._visualizer.newFrame.connect(self._onNewFrame)

        w, h = self._cameraDevice.frameSize
        self.setMinimumSize(w, h)
        self.setMaximumSize(w, h)

    @QtCore.pyqtSlot(cv2.iplimage)
    def _onNewFrame(self, frame):
        self.newFrame.emit(self.frame)
        self.update()

    def changeEvent(self, e):
        if e.type() == QtCore.QEvent.EnabledChange:
            if self.isEnabled():
                self._visualizer.newFrame.connect(self._onNewFrame)
            else:
                self._visualizer.newFrame.disconnect(self._onNewFrame)

    def paintEvent(self, e):
        if self.frame is None:
            return
        painter = QtGui.QPainter(self)
        painter.drawImage(QtCore.QPoint(0, 0), cv2.OpenCVQImage(self.frame))