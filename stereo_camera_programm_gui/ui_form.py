# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'form.ui'
##
## Created by: Qt User Interface Compiler version 6.7.0
##
## WARNING! All changes made in this file will be lost when recompiling UI file!
################################################################################

from PySide6.QtCore import (QCoreApplication, QDate, QDateTime, QLocale,
    QMetaObject, QObject, QPoint, QRect,
    QSize, QTime, QUrl, Qt)
from PySide6.QtGui import (QBrush, QColor, QConicalGradient, QCursor,
    QFont, QFontDatabase, QGradient, QIcon,
    QImage, QKeySequence, QLinearGradient, QPainter,
    QPalette, QPixmap, QRadialGradient, QTransform)
from PySide6.QtWidgets import (QApplication, QCheckBox, QLabel, QPushButton,
    QSizePolicy, QSlider, QTabWidget, QTextEdit,
    QWidget)

class Ui_Widget(object):
    def setupUi(self, Widget):
        if not Widget.objectName():
            Widget.setObjectName(u"Widget")
        Widget.resize(800, 600)
        self.tabWidget = QTabWidget(Widget)
        self.tabWidget.setObjectName(u"tabWidget")
        self.tabWidget.setGeometry(QRect(0, 0, 801, 601))
        self.tab = QWidget()
        self.tab.setObjectName(u"tab")
        self.del_bottom = QCheckBox(self.tab)
        self.del_bottom.setObjectName(u"del_bottom")
        self.del_bottom.setGeometry(QRect(540, 220, 78, 22))
        self.del_upper = QCheckBox(self.tab)
        self.del_upper.setObjectName(u"del_upper")
        self.del_upper.setGeometry(QRect(540, 260, 151, 22))
        self.del_dorn = QCheckBox(self.tab)
        self.del_dorn.setObjectName(u"del_dorn")
        self.del_dorn.setGeometry(QRect(540, 300, 111, 22))
        self.rest = QCheckBox(self.tab)
        self.rest.setObjectName(u"rest")
        self.rest.setGeometry(QRect(540, 340, 121, 22))
        self.rog = QCheckBox(self.tab)
        self.rog.setObjectName(u"rog")
        self.rog.setGeometry(QRect(540, 180, 221, 22))
        self.label_2 = QLabel(self.tab)
        self.label_2.setObjectName(u"label_2")
        self.label_2.setGeometry(QRect(640, 10, 141, 81))
        self.label_2.setTextFormat(Qt.AutoText)
        self.label_2.setPixmap(QPixmap(u"../iph_logo.png"))
        self.label_2.setScaledContents(True)
        self.choose_pc = QPushButton(self.tab)
        self.choose_pc.setObjectName(u"choose_pc")
        self.choose_pc.setGeometry(QRect(140, 240, 211, 81))
        self.refresh = QPushButton(self.tab)
        self.refresh.setObjectName(u"refresh")
        self.refresh.setGeometry(QRect(540, 380, 121, 51))
        self.tabWidget.addTab(self.tab, "")
        self.tab_2 = QWidget()
        self.tab_2.setObjectName(u"tab_2")
        self.pushButton = QPushButton(self.tab_2)
        self.pushButton.setObjectName(u"pushButton")
        self.pushButton.setGeometry(QRect(20, 520, 80, 24))
        self.explain_data = QTextEdit(self.tab_2)
        self.explain_data.setObjectName(u"explain_data")
        self.explain_data.setGeometry(QRect(470, 260, 321, 271))
        self.label = QLabel(self.tab_2)
        self.label.setObjectName(u"label")
        self.label.setGeometry(QRect(640, 10, 141, 81))
        self.label.setTextFormat(Qt.AutoText)
        self.label.setPixmap(QPixmap(u"../iph_logo.png"))
        self.label.setScaledContents(True)
        self.label_3 = QLabel(self.tab_2)
        self.label_3.setObjectName(u"label_3")
        self.label_3.setGeometry(QRect(60, 40, 251, 251))
        self.label_3.setPixmap(QPixmap(u"../Daten/crop/crop1.png"))
        self.label_3.setScaledContents(False)
        self.x_koor = QSlider(self.tab_2)
        self.x_koor.setObjectName(u"x_koor")
        self.x_koor.setGeometry(QRect(70, 330, 160, 16))
        self.x_koor.setOrientation(Qt.Horizontal)
        self.horizontalSlider_2 = QSlider(self.tab_2)
        self.horizontalSlider_2.setObjectName(u"horizontalSlider_2")
        self.horizontalSlider_2.setGeometry(QRect(70, 370, 160, 16))
        self.horizontalSlider_2.setOrientation(Qt.Horizontal)
        self.tabWidget.addTab(self.tab_2, "")

        self.retranslateUi(Widget)

        self.tabWidget.setCurrentIndex(1)


        QMetaObject.connectSlotsByName(Widget)
    # setupUi

    def retranslateUi(self, Widget):
        Widget.setWindowTitle(QCoreApplication.translate("Widget", u"Widget", None))
        self.del_bottom.setText(QCoreApplication.translate("Widget", u"CheckBox", None))
        self.del_upper.setText(QCoreApplication.translate("Widget", u"Oberschicht entfernen", None))
        self.del_dorn.setText(QCoreApplication.translate("Widget", u"Dorn entfernen", None))
        self.rest.setText(QCoreApplication.translate("Widget", u"Rest entfernen", None))
        self.rog.setText(QCoreApplication.translate("Widget", u"Zugeschnitten auf Region of Interrest", None))
        self.label_2.setText("")
        self.choose_pc.setText(QCoreApplication.translate("Widget", u"Auswahl der Punktwolke", None))
        self.refresh.setText(QCoreApplication.translate("Widget", u"Anwenden", None))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab), QCoreApplication.translate("Widget", u"Punktwolke", None))
        self.pushButton.setText(QCoreApplication.translate("Widget", u"Start", None))
        self.explain_data.setHtml(QCoreApplication.translate("Widget", u"<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
"<html><head><meta name=\"qrichtext\" content=\"1\" /><meta charset=\"utf-8\" /><style type=\"text/css\">\n"
"p, li { white-space: pre-wrap; }\n"
"hr { height: 1px; border-width: 0; }\n"
"li.unchecked::marker { content: \"\\2610\"; }\n"
"li.checked::marker { content: \"\\2612\"; }\n"
"</style></head><body style=\" font-family:'Segoe UI'; font-size:9pt; font-weight:400; font-style:normal;\">\n"
"<p style=\" margin-top:12px; margin-bottom:12px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">Zur Datenverarbeitung der Aufnahmen:</p>\n"
"<p style=\" margin-top:12px; margin-bottom:12px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">-Alle zu verarbeiteten Daten m\u00fcssen in dem Ordner Messungen liegen</p>\n"
"<p style=\" margin-top:12px; margin-bottom:12px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">-Die Punktwolken m\u00fcssen in"
                        " den 3d-Ordner gepackt werden</p>\n"
"<p style=\" margin-top:12px; margin-bottom:12px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">-Die Bilder d\u00fcrfen in beliebig benannte Ordner innerhalb des Messungsordners liegen</p>\n"
"<p style=\" margin-top:12px; margin-bottom:12px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">-vor dem Verarbeiteten der Daten muss mit den x und y Koordinaten das Schmiedeteil in die Mitte des Bildes gebracht werden</p>\n"
"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><br /></p></body></html>", None))
        self.label.setText("")
        self.label_3.setText("")
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab_2), QCoreApplication.translate("Widget", u"Datenverarbeitung", None))
    # retranslateUi

