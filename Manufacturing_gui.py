from PyQt5.QtWidgets import *
from PyQt5 import uic
import os
from PyQt5 import QtGui
import sys
# python Ui Directories
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TEST_UI = uic.loadUiType(BASE_DIR + r'\Manufacturing_UI.ui')[0]

# Content

# #### MAIN ###

class TEST_window(QMainWindow, TEST_UI):
    def __init__(self):
        super().__init__()
        self.setupUi(self)
        self.setWindowTitle("제조시스템")
        self.setWindowIcon(QtGui.QIcon(BASE_DIR+r"\yun.png"))
        self.FILEPATH.clicked.connect(self.FILE_func)
        self.DAWN.clicked.connect(self.DAWN_func)
        self.NORMAL.clicked.connect(self.NORMAL_func)
        self.DIRECT.clicked.connect(self.DIRECT_func)
        self.insertbtn.clicked.connect(self.insert_func)
        self.clearbtn.clicked.connect(self.clear_func)
        
        self.START.setEnabled(False)
        self.START.clicked.connect(self.Test_func)
        self.STOP.clicked.connect(self.STOP_func)
        
        self.dawn_check.setEnabled(False)
        self.normal_check.setEnabled(False)
        self.direct_check.setEnabled(False)
        
        self.msg = QMessageBox()
        self.show()
        
    def FILE_func(self):
        global file_path, file_name
        file_path = QFileDialog.getOpenFileName(self)[0]
        file_name = os.path.basename(file_path)
        self.File_label.setText(file_name)
        print(file_path)
    
    def DAWN_func(self):
        global sheet_name
        sheet_name = '요청사항표-새벽배송'
        self.dawn_check.toggle()
        self.normal_check.setChecked(False)
        self.direct_check.setChecked(False)
        
        print(sheet_name)
        
    def NORMAL_func(self):
        global sheet_name
        sheet_name = '요청사항표-일반배송'
        self.normal_check.toggle()
        self.dawn_check.setChecked(False)
        self.direct_check.setChecked(False)
        print(sheet_name)
        
    def DIRECT_func(self):
        global sheet_name
        sheet_name = '요청사항표-직접배송'
        self.direct_check.toggle()
        self.dawn_check.setChecked(False)
        self.normal_check.setChecked(False)
        print(sheet_name)
        
    def insert_func(self):        
        import datetime
        today = datetime.datetime.today().strftime('%Y-%m-%d-%A %H:%M:%S')
        parameter_str = f'현재시각 : {today} \n파일이름 : {file_name} \n배송종류 : {sheet_name}'
        self.Parameters.setText(parameter_str)
        self.START.setEnabled(True)
        
    def clear_func(self):
        global file_name, sheet_name
        file_name = ''
        sheet_name = ''
        self.File_label.setText('')
        self.Parameters.setText('초기화되었습니다.\n다시 선택해주시기 바랍니다.')
        self.START.setEnabled(False)
    
    def STOP_func(self):
        self.msg.setWindowTitle('종료')
        self.setWindowIcon(QtGui.QIcon(BASE_DIR+r"\yun.png"))
        self.msg.setIcon(QMessageBox.Warning)
        self.msg.setText("제조시스템을 종료하겠습니다.")
        self.msg.setStandardButtons(QMessageBox.Yes | QMessageBox.Cancel) 
        reply = self.msg.exec()
        if reply == QMessageBox.Yes:    
            import sys
            sys.exit()           
        else:
            pass
            
        
    def Test_func(self):
        try:
            
            from Functions import Manufacturing_function
            WEIGHTS = Manufacturing_function.WEIGHTS
            # from pathlib import Path
            # import sys
            # import os
            # FILE = Path(os.getcwd()+'/Functions').resolve()
            # WEIGHT_ROOT = FILE
            # ROOT = FILE
            # WEIGHTS = WEIGHT_ROOT / 'weights'
            # if str(ROOT) not in sys.path:
            #     sys.path.append(str(ROOT))  # add ROOT to PATH
            # if str(ROOT / 'yolov5') not in sys.path:
            #     sys.path.append(str(ROOT / 'yolov5'))  # add yolov5 ROOT to PATH
            # if str(ROOT / 'trackers' / 'strong_sort') not in sys.path:
            #     sys.path.append(str(ROOT / 'trackers' / 'strong_sort'))  # add strong_sort ROOT to PATH
            # if str(ROOT / 'trackers' / 'ocsort') not in sys.path:
            #     sys.path.append(str(ROOT / 'trackers' / 'ocsort'))  # add strong_sort ROOT to PATH
            # if str(ROOT / 'trackers' / 'strong_sort' / 'deep' / 'reid' / 'torchreid') not in sys.path:
            #     sys.path.append(str(ROOT / 'trackers' / 'strong_sort' / 'deep' / 'reid' / 'torchreid'))  # add strong_sort ROOT to PATH
            # ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative
            Manufacturing_function.run(
                source = 'webcams.txt',
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
                save_vid=True,  # save confidences in --save-txt labels
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
        except ValueError as vle:
            self.msg.setWindowTitle('ERROR')
            self.setWindowIcon(QtGui.QIcon(BASE_DIR+r"\yun.png"))
            self.msg.setIcon(QMessageBox.Information)
            self.msg.setText(f"{vle}\n엑셀 및 배송 내용이 없습니다.")
            retval = self.msg.exec()
            
            self.START.setEnabled(False)
        except Exception as e:
            self.msg.setWindowTitle('DEV.ERROR')
            self.setWindowIcon(QtGui.QIcon(BASE_DIR+r"\yun.png"))
            self.msg.setIcon(QMessageBox.Critical)
            if 'Failed to open' in str(e):
                webcam_error = f"{e}\n웹 캠 연결이 되지 않았습니다.\n다시확인 바랍니다."
                self.msg.setText(f"{webcam_error}")
            else:
                self.msg.setText(f"{e}")
            retval = self.msg.exec()
            self.START.setEnabled(False)
if __name__ == '__main__':
    app = QApplication(sys.argv)
    Window = TEST_window()
    Window.show()
    app.exec()
