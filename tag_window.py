# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'Tag_dialog_box.ui'
#
# Created by: PyQt5 UI code generator 5.6
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets

class Ui_Dialog(object):
    def setupUi(self, Dialog):
        Dialog.setObjectName("Dialog")
        Dialog.resize(397, 344)
        self.button_box = QtWidgets.QDialogButtonBox(Dialog)
        self.button_box.setGeometry(QtCore.QRect(290, 20, 81, 241))
        self.button_box.setOrientation(QtCore.Qt.Vertical)
        self.button_box.setStandardButtons(QtWidgets.QDialogButtonBox.Apply)
        self.button_box.setCenterButtons(False)
        self.button_box.setObjectName("button_box")
        self.start_time = QtWidgets.QLineEdit(Dialog)
        self.start_time.setGeometry(QtCore.QRect(30, 20, 113, 20))
        self.start_time.setObjectName("start_time")
        self.end_time = QtWidgets.QLineEdit(Dialog)
        self.end_time.setGeometry(QtCore.QRect(30, 40, 113, 20))
        self.end_time.setObjectName("end_time")
        self.classification = QtWidgets.QLineEdit(Dialog)
        self.classification.setGeometry(QtCore.QRect(30, 60, 113, 20))
        self.classification.setObjectName("classification")
        self.electrodes = QtWidgets.QLineEdit(Dialog)
        self.electrodes.setGeometry(QtCore.QRect(30, 80, 113, 20))
        self.electrodes.setObjectName("electrodes")
        self.tableWidget = QtWidgets.QTableWidget(Dialog)
        self.tableWidget.setGeometry(QtCore.QRect(30, 150, 351, 192))
        self.tableWidget.setObjectName("tableWidget")
        self.tableWidget.setColumnCount(7)
        self.tableWidget.setRowCount(0)
        item = QtWidgets.QTableWidgetItem()
        self.tableWidget.setHorizontalHeaderItem(0, item)
        item = QtWidgets.QTableWidgetItem()
        self.tableWidget.setHorizontalHeaderItem(1, item)
        item = QtWidgets.QTableWidgetItem()
        self.tableWidget.setHorizontalHeaderItem(2, item)
        item = QtWidgets.QTableWidgetItem()
        self.tableWidget.setHorizontalHeaderItem(3, item)
        item = QtWidgets.QTableWidgetItem()
        self.tableWidget.setHorizontalHeaderItem(4, item)
        item = QtWidgets.QTableWidgetItem()
        self.tableWidget.setHorizontalHeaderItem(5, item)
        item = QtWidgets.QTableWidgetItem()
        self.tableWidget.setHorizontalHeaderItem(6, item)
        self.stim_number = QtWidgets.QLineEdit(Dialog)
        self.stim_number.setGeometry(QtCore.QRect(30, 100, 113, 20))
        self.stim_number.setObjectName("stim_number")
        self.stim_time = QtWidgets.QLineEdit(Dialog)
        self.stim_time.setGeometry(QtCore.QRect(30, 120, 113, 20))
        self.stim_time.setObjectName("stim_time")
        self.Delete_row = QtWidgets.QPushButton(Dialog)
        self.Delete_row.setGeometry(QtCore.QRect(290, 50, 75, 23))
        self.Delete_row.setObjectName("Delete_row")
        self.Export = QtWidgets.QPushButton(Dialog)
        self.Export.setGeometry(QtCore.QRect(290, 80, 75, 23))
        self.Export.setObjectName("Export")
        self.load_csv = QtWidgets.QPushButton(Dialog)
        self.load_csv.setGeometry(QtCore.QRect(290, 110, 75, 23))
        self.load_csv.setObjectName("load_csv")

        self.retranslateUi(Dialog)
        self.button_box.accepted.connect(Dialog.accept)
        self.button_box.rejected.connect(Dialog.reject)
        QtCore.QMetaObject.connectSlotsByName(Dialog)

    def retranslateUi(self, Dialog):
        _translate = QtCore.QCoreApplication.translate
        Dialog.setWindowTitle(_translate("Dialog", "Dialog"))
        self.start_time.setText(_translate("Dialog", "Start Time"))
        self.end_time.setText(_translate("Dialog", "End Time"))
        self.classification.setText(_translate("Dialog", "Classification"))
        self.electrodes.setText(_translate("Dialog", "Electrodes"))
        item = self.tableWidget.horizontalHeaderItem(0)
        item.setText(_translate("Dialog", "ME number"))
        item = self.tableWidget.horizontalHeaderItem(1)
        item.setText(_translate("Dialog", "Type"))
        item = self.tableWidget.horizontalHeaderItem(2)
        item.setText(_translate("Dialog", "Video_time [s]"))
        item = self.tableWidget.horizontalHeaderItem(3)
        item.setText(_translate("Dialog", "Length [s]"))
        item = self.tableWidget.horizontalHeaderItem(4)
        item.setText(_translate("Dialog", "New Column"))
        item = self.tableWidget.horizontalHeaderItem(5)
        item.setText(_translate("Dialog", "Stimulus number"))
        item = self.tableWidget.horizontalHeaderItem(6)
        item.setText(_translate("Dialog", "Stimulus time [s]"))
        self.stim_number.setText(_translate("Dialog", "Stimulus Number"))
        self.stim_time.setText(_translate("Dialog", "Stimulus Time"))
        self.Delete_row.setText(_translate("Dialog", "Delete row"))
        self.Export.setText(_translate("Dialog", "Export "))
        self.load_csv.setText(_translate("Dialog", "Load "))

