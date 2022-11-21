import sys
from PyQt5.QtWidgets import *
from PyQt5 import uic
import os
import subprocess
# from PyQt5 import QtCore
# from PyQt5 import QtGui

# python Ui Directories
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TEST_UI = uic.loadUiType(BASE_DIR + r'\test.ui')[0]

# PYQT

# #### MAIN ###


# class RFID_window(QDialog, QWidget, RFID_ui):
class TEST_window(QMainWindow, TEST_UI):
    def __init__(self):
        # super(RFID_window, self).__init__()
        super().__init__()
        self.setupUi(self)
        self.setWindowTitle("TEST")
        self.pushButton.clicked.connect(self.Test_func)
        self.show()

    def Test_func(self):
        # os.system(
        #     "python track_final.py --device 0 --yolo-weights weights/yolov5m.pt --source 0 --file-path test.xlsx --sheet-name 요청사항표-새벽배송 --show-vid"
        # )
        result = subprocess.run(["track_final.py", "--device 0--yolo-weights weights/yolov5m.pt --source 0 --file-path test.xlsx --sheet-name 요청사항표-새벽배송 --show-vid"], stdout=subprocess.PIPE, text=True)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    Window = TEST_window()
    Window.show()
    app.exec()
