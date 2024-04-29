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
from PySide6.QtWidgets import (QApplication, QCheckBox, QLabel, QLineEdit,
    QListWidget, QListWidgetItem, QPushButton, QSizePolicy,
    QTabWidget, QTextEdit, QWidget)

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
        self.del_bottom.setGeometry(QRect(490, 220, 271, 21))
        self.del_upper = QCheckBox(self.tab)
        self.del_upper.setObjectName(u"del_upper")
        self.del_upper.setGeometry(QRect(490, 260, 151, 22))
        self.del_dorn = QCheckBox(self.tab)
        self.del_dorn.setObjectName(u"del_dorn")
        self.del_dorn.setGeometry(QRect(490, 300, 111, 22))
        self.rest = QCheckBox(self.tab)
        self.rest.setObjectName(u"rest")
        self.rest.setGeometry(QRect(490, 340, 301, 22))
        self.rog = QCheckBox(self.tab)
        self.rog.setObjectName(u"rog")
        self.rog.setGeometry(QRect(490, 180, 221, 22))
        self.iph_logo_2 = QLabel(self.tab)
        self.iph_logo_2.setObjectName(u"iph_logo_2")
        self.iph_logo_2.setGeometry(QRect(640, 10, 141, 81))
        self.iph_logo_2.setTextFormat(Qt.AutoText)
        self.iph_logo_2.setPixmap(QPixmap(u"../iph_logo.png"))
        self.iph_logo_2.setScaledContents(True)
        self.refresch = QPushButton(self.tab)
        self.refresch.setObjectName(u"refresch")
        self.refresch.setGeometry(QRect(110, 70, 211, 81))
        self.execut_pc_pro = QPushButton(self.tab)
        self.execut_pc_pro.setObjectName(u"execut_pc_pro")
        self.execut_pc_pro.setGeometry(QRect(540, 380, 121, 51))
        self.pc_file_list = QListWidget(self.tab)
        self.pc_file_list.setObjectName(u"pc_file_list")
        self.pc_file_list.setGeometry(QRect(90, 160, 256, 351))
        self.choosen_pc_file = QLabel(self.tab)
        self.choosen_pc_file.setObjectName(u"choosen_pc_file")
        self.choosen_pc_file.setGeometry(QRect(100, 520, 241, 31))
        self.tabWidget.addTab(self.tab, "")
        self.tab_2 = QWidget()
        self.tab_2.setObjectName(u"tab_2")
        self.start_bt = QPushButton(self.tab_2)
        self.start_bt.setObjectName(u"start_bt")
        self.start_bt.setGeometry(QRect(20, 520, 80, 24))
        self.explain_data = QTextEdit(self.tab_2)
        self.explain_data.setObjectName(u"explain_data")
        self.explain_data.setGeometry(QRect(470, 260, 321, 271))
        self.iph_logo_1 = QLabel(self.tab_2)
        self.iph_logo_1.setObjectName(u"iph_logo_1")
        self.iph_logo_1.setGeometry(QRect(640, 10, 141, 81))
        self.iph_logo_1.setTextFormat(Qt.AutoText)
        self.iph_logo_1.setPixmap(QPixmap(u"../iph_logo.png"))
        self.iph_logo_1.setScaledContents(True)
        self.label_3 = QLabel(self.tab_2)
        self.label_3.setObjectName(u"label_3")
        self.label_3.setGeometry(QRect(60, 40, 251, 251))
        self.label_3.setPixmap(QPixmap(u"../Daten/crop/crop1.png"))
        self.label_3.setScaledContents(False)
        self.crop_img = QLabel(self.tab_2)
        self.crop_img.setObjectName(u"crop_img")
        self.crop_img.setEnabled(True)
        self.crop_img.setGeometry(QRect(20, 60, 281, 241))
        self.crop_img.setTextFormat(Qt.AutoText)
        self.crop_img.setPixmap(QPixmap(u"../../stereocamera_mesurment/Daten/crop/crop1.png"))
        self.crop_img.setScaledContents(True)
        self.x_crop_value = QLabel(self.tab_2)
        self.x_crop_value.setObjectName(u"x_crop_value")
        self.x_crop_value.setGeometry(QRect(30, 330, 49, 16))
        self.y_crop_value = QLabel(self.tab_2)
        self.y_crop_value.setObjectName(u"y_crop_value")
        self.y_crop_value.setGeometry(QRect(30, 360, 49, 16))
        self.x_input_value = QLineEdit(self.tab_2)
        self.x_input_value.setObjectName(u"x_input_value")
        self.x_input_value.setGeometry(QRect(70, 330, 113, 24))
        self.y_input_value = QLineEdit(self.tab_2)
        self.y_input_value.setObjectName(u"y_input_value")
        self.y_input_value.setGeometry(QRect(70, 360, 113, 24))
        self.tabWidget.addTab(self.tab_2, "")

        self.retranslateUi(Widget)

        self.tabWidget.setCurrentIndex(0)


        QMetaObject.connectSlotsByName(Widget)
    # setupUi

    def retranslateUi(self, Widget):
        Widget.setWindowTitle(QCoreApplication.translate("Widget", u"GUI", None))
        self.del_bottom.setText(QCoreApplication.translate("Widget", u"Untergrund in der Region of Interest entfernen", None))
        self.del_upper.setText(QCoreApplication.translate("Widget", u"Oberschicht entfernen", None))
        self.del_dorn.setText(QCoreApplication.translate("Widget", u"Dorn entfernen", None))
        self.rest.setText(QCoreApplication.translate("Widget", u"Dorn und Obereschicht behalten und Rest entfernen", None))
        self.rog.setText(QCoreApplication.translate("Widget", u"Zugeschnitten auf Region of Interrest", None))
        self.iph_logo_2.setText("")
        self.refresch.setText(QCoreApplication.translate("Widget", u"Aktualesieren", None))
        self.execut_pc_pro.setText(QCoreApplication.translate("Widget", u"Anwenden", None))
        self.choosen_pc_file.setText(QCoreApplication.translate("Widget", u"Ausgew\u00e4hlt ist 176.csv", None))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab), QCoreApplication.translate("Widget", u"Punktwolke", None))
        self.start_bt.setText(QCoreApplication.translate("Widget", u"Start", None))
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
        self.iph_logo_1.setText("")
        self.label_3.setText("")
        self.crop_img.setText("")
        self.x_crop_value.setText(QCoreApplication.translate("Widget", u"x:650", None))
        self.y_crop_value.setText(QCoreApplication.translate("Widget", u"y: 600", None))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab_2), QCoreApplication.translate("Widget", u"Datenverarbeitung", None))
    # retranslateUi

