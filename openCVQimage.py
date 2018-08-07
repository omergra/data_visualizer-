import cv2

from PyQt5 import QtGui


class OpenCVQImage(QtGui.QImage):

    def __init__(self, opencvBgrImg):
        # it's assumed the image is in BGR format
        return cv2.cvtColor(opencvBgrImg[1],cv2.COLOR_BGR2RGB)

