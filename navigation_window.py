# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'navigation_window.ui'
#
# Created by: PyQt5 UI code generator 5.6
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets

class Ui_Dialog_navigation(object):
    def setupUi(self, Dialog_navigation):
        Dialog_navigation.setObjectName("Dialog_navigation")
        Dialog_navigation.resize(400, 300)
        self.buttonBox = QtWidgets.QDialogButtonBox(Dialog_navigation)
        self.buttonBox.setGeometry(QtCore.QRect(290, 20, 81, 241))
        self.buttonBox.setOrientation(QtCore.Qt.Vertical)
        self.buttonBox.setStandardButtons(QtWidgets.QDialogButtonBox.Cancel|QtWidgets.QDialogButtonBox.Ok)
        self.buttonBox.setObjectName("buttonBox")

        self.retranslateUi(Dialog_navigation)
        self.buttonBox.accepted.connect(Dialog_navigation.accept)
        self.buttonBox.rejected.connect(Dialog_navigation.reject)
        QtCore.QMetaObject.connectSlotsByName(Dialog_navigation)

    def retranslateUi(self, Dialog_navigation):
        _translate = QtCore.QCoreApplication.translate
        Dialog_navigation.setWindowTitle(_translate("Dialog_navigation", "Dialog"))


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    Dialog_navigation = QtWidgets.QDialog()
    ui = Ui_Dialog_navigation()
    ui.setupUi(Dialog_navigation)
    Dialog_navigation.show()
    sys.exit(app.exec_())

