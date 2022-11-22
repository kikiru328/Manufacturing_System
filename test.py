import sys
from PyQt5.QtWidgets import *
from PyQt5 import uic
import os

# from PyQt5 import QtCore
# from PyQt5 import QtGui

# python Ui Directories
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TEST_UI = uic.loadUiType(BASE_DIR + r'\test.ui')[0]

# Content

# #### MAIN ###


# class RFID_window(QDialog, QWidget, RFID_ui):
class TEST_window(QMainWindow, TEST_UI):
    def __init__(self):
        # super(RFID_window, self).__init__()
        super().__init__()
        self.setupUi(self)
        self.setWindowTitle("TEST")
        self.START.clicked.connect(self.Test_func)
        self.FILEPATH.clicked.connect(self.FILE_func)
        self.DAWN.clicked.connect(self.DAWN_func)
        self.NORMAL.clicked.connect(self.NORMAL_func)
        self.DIRECT.clicked.connect(self.DIRECT_func)
        self.STOP.clicked.connect(self.STOP_func)
        
        self.show()
    def FILE_func(self):
        global file_path
        file_path = QFileDialog.getOpenFileName(self)[0]
        print(file_path)
    
    def DAWN_func(self):
        global sheet_name
        sheet_name = '요청사항표-새벽배송'
        print(sheet_name)
        
    def NORMAL_func(self):
        global sheet_name
        sheet_name = '요청사항표-일반배송'
        print(sheet_name)
        
    def DIRECT_func(self):
        global sheet_name
        sheet_name = '요청사항표-직접배송'
        print(sheet_name)
        
                
    def Test_func(self):
        import track_for_test_copy
        from pathlib import Path
        import sys
        import os
        FILE = Path(os.getcwd()).resolve()
        ROOT = FILE.parents[0] 
        # ROOT = os.getcwd()
        WEIGHTS = ROOT / 'weights'
        if str(ROOT) not in sys.path:
            sys.path.append(str(ROOT))  # add ROOT to PATH
        if str(ROOT / 'yolov5') not in sys.path:
            sys.path.append(str(ROOT / 'yolov5'))  # add yolov5 ROOT to PATH
        if str(ROOT / 'trackers' / 'strong_sort') not in sys.path:
            sys.path.append(str(ROOT / 'trackers' / 'strong_sort'))  # add strong_sort ROOT to PATH
        if str(ROOT / 'trackers' / 'ocsort') not in sys.path:
            sys.path.append(str(ROOT / 'trackers' / 'ocsort'))  # add strong_sort ROOT to PATH
        if str(ROOT / 'trackers' / 'strong_sort' / 'deep' / 'reid' / 'torchreid') not in sys.path:
            sys.path.append(str(ROOT / 'trackers' / 'strong_sort' / 'deep' / 'reid' / 'torchreid'))  # add strong_sort ROOT to PATH
        ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative
        track_for_test_copy.run(
            source = '0',
            yolo_weights=WEIGHTS / 'yolov5m.pt',
            reid_weights=WEIGHTS / 'osnet_x0_25_msmt17.pt',
            tracking_method='strongsort',
            imgsz=(640, 640),  # inference size (height, width)
            conf_thres=0.25,  # confidence threshold
            iou_thres=0.45,  # NMS IOU threshold
            max_det=1000,  # maximum detections per image
            device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu 
            show_vid=True,  # show results
            save_txt=False,  # save results to *.txt
            save_conf=False,  # save confidences in --save-txt labels
            save_crop=False,  # save cropped prediction boxes
            save_vid=False,  # save confidences in --save-txt labels
            nosave=False,  # do not save images/videos
            classes=None,  # filter by class: --class 0, or --class 0 2 3
            agnostic_nms=False,  # class-agnostic NMS
            augment=False,  # augmented inference
            visualize=False,  # visualize features
            update=False,  # update all models
            project='./runs/track',
            name='exp',  # save results to project/name
            exist_ok=False,  # existing project/name ok, do not increment
            line_thickness=2,  # bounding box thickness (pixels)
            hide_labels=False,  # hide labels
            hide_conf=False,  # hide confidences
            hide_class=False,  # hide IDs
            half=False,  # use FP16 half-precision inference
            dnn=False,  # use OpenCV DNN for ONNX inference
            eval=False,  # run multi-gpu eval,
            file_path=file_path,
            sheet_name=sheet_name    
        )

    def STOP_func(self):
        import sys
        sys.exit()
        
if __name__ == '__main__':
    app = QApplication(sys.argv)
    Window = TEST_window()
    Window.show()
    app.exec()
