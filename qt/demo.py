# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'demo.ui',
# licensing of 'demo.ui' applies.
#
# Created: Fri May  8 10:11:12 2020
#      by: pyside2-uic  running on PySide2 5.13.2
#
# WARNING! All changes made in this file will be lost!

from PySide2 import QtCore, QtGui, QtWidgets

class Ui_Form(object):
    def setupUi(self, Form):
        Form.setObjectName("Form")
        Form.resize(1166, 670)
        self.originalLabel = QtWidgets.QLabel(Form)
        self.originalLabel.setGeometry(QtCore.QRect(50, 70, 480, 480))
        self.originalLabel.setAutoFillBackground(False)
        self.originalLabel.setText("")
        self.originalLabel.setObjectName("originalLabel")
        self.segLabel = QtWidgets.QLabel(Form)
        self.segLabel.setGeometry(QtCore.QRect(590, 70, 480, 480))
        self.segLabel.setAutoFillBackground(False)
        self.segLabel.setText("")
        self.segLabel.setObjectName("segLabel")
        self.inputButton = QtWidgets.QPushButton(Form)
        self.inputButton.setGeometry(QtCore.QRect(290, 590, 150, 30))
        self.inputButton.setObjectName("inputButton")
        self.outputButton = QtWidgets.QPushButton(Form)
        self.outputButton.setGeometry(QtCore.QRect(830, 590, 150, 30))
        self.outputButton.setObjectName("outputButton")

        self.retranslateUi(Form)
        QtCore.QObject.connect(self.inputButton, QtCore.SIGNAL("clicked()"), Form.input)
        QtCore.QObject.connect(self.outputButton, QtCore.SIGNAL("clicked()"), Form.output)
        QtCore.QMetaObject.connectSlotsByName(Form)

    def retranslateUi(self, Form):
        Form.setWindowTitle(QtWidgets.QApplication.translate("Form", "视网膜血管分割", None, -1))
        self.inputButton.setText(QtWidgets.QApplication.translate("Form", "输入图像", None, -1))
        self.outputButton.setText(QtWidgets.QApplication.translate("Form", "保存结果", None, -1))

