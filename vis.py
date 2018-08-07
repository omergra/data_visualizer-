# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'visualizer.ui'
#
# Created by: PyQt5 UI code generator 5.9.2
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(800, 801)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.splitter = QtWidgets.QSplitter(self.centralwidget)
        self.splitter.setGeometry(QtCore.QRect(10, 30, 791, 721))
        self.splitter.setOrientation(QtCore.Qt.Vertical)
        self.splitter.setObjectName("splitter")
        self.graphicsView_for_video = QtWidgets.QGraphicsView(self.splitter)
        self.graphicsView_for_video.setObjectName("graphicsView_for_video")
        self.graphicsView = GraphicsWindow(self.splitter)
        self.graphicsView.setObjectName("graphicsView")
        self.horizontalSlider = QtWidgets.QSlider(self.splitter)
        self.horizontalSlider.setOrientation(QtCore.Qt.Horizontal)
        self.horizontalSlider.setObjectName("horizontalSlider")
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 800, 21))
        self.menubar.setObjectName("menubar")
        self.menuData_Visualizer = QtWidgets.QMenu(self.menubar)
        self.menuData_Visualizer.setObjectName("menuData_Visualizer")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)
        self.actionLoad_Video = QtWidgets.QAction(MainWindow)
        self.actionLoad_Video.setObjectName("actionLoad_Video")
        self.actionLoad_data = QtWidgets.QAction(MainWindow)
        self.actionLoad_data.setObjectName("actionLoad_data")
        self.menuData_Visualizer.addAction(self.actionLoad_Video)
        self.menuData_Visualizer.addAction(self.actionLoad_data)
        self.menubar.addAction(self.menuData_Visualizer.menuAction())

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.menuData_Visualizer.setTitle(_translate("MainWindow", "File"))
        self.actionLoad_Video.setText(_translate("MainWindow", "Load Video"))
        self.actionLoad_data.setText(_translate("MainWindow", "Load data"))

from pyqtgraph import GraphicsWindow

if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())

