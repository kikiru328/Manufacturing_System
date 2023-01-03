from PyQt5.QtWidgets import *
from PyQt5 import uic
from PyQt5 import QtGui
import os
import sys
import win32com.client
import pythoncom
import time
# python Ui Directories
BASE_DIR =os.path.dirname(os.path.abspath(__file__))
install_UI = uic.loadUiType(BASE_DIR + r'\install_UI.ui')[0]
sys.path.append(BASE_DIR + r'\Functions')
print("설치시스템 실행됩니다.")
# Content

# #### MAIN ###

class install_window(QMainWindow, install_UI):
    def __init__(self):
        super().__init__()
        self.setupUi(self)
        self.setWindowTitle("설치프로그램")
        self.setWindowIcon(QtGui.QIcon(BASE_DIR+r"\yun.png"))
        self.miniconda_btn.clicked.connect(self.miniconda_btn_func)
        self.add_btn.clicked.connect(self.add_btn_func)
        self.install_btn.clicked.connect(self.install_btn_func)
        self.msg = QMessageBox()
        self.show()
        
    def miniconda_btn_func(self):
        path = os.path.realpath('Miniconda')
        os.startfile(path)
        
    def add_btn_func(self):
        os.system('conda create -n manufacturing python=3.8 -y')
        os.system('Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser')
        os.system('conda activate manufacturing')
        time.sleep(5)
        os.system(f"pip install -r {os.path.realpath('requirements.txt')}")
        os.system('conda install -c conda-forge lap -y') 
        os.system('pip install cython-bbox-windows')
        os.system('pip install screeninfo')
        print('완료')

        self.setWindowIcon(QtGui.QIcon(BASE_DIR+r"\yun.png"))
        self.msg.setIcon(QMessageBox.Information)
        self.msg.setText(f"추가 설치가 완료되었습니다.")
        
    def install_btn_func(self):
        gui_path = os.path.realpath('../Manufacturing_gui.py')
        icon_path = os.path.realpath('yun.ico')
        # python_path = sys.exec_prefix + '/python.exe'
        python_path = os.path.expanduser('~') + "\\miniconda3\python.exe"
        with open(os.getcwd()+r'\제조시스템.bat','w') as bat_file:
            bat_file.write(
                f"start {python_path} {gui_path} %*" +"\npause"
            )
            
        # bat_file = mbs.findFile('test_bat.bat')
        bat_file = os.path.realpath('제조시스템.bat')
        desktop_path = os.path.expanduser('~') + "\\Desktop\\"
        pythoncom.CoInitialize() 
        desktop_path = os.path.expanduser('~') + "\\Desktop\\"
        path = os.path.join(desktop_path, '제조시스템.lnk')
        target = bat_file
        icon = icon_path

        shell = win32com.client.Dispatch("WScript.Shell")
        shortcut = shell.CreateShortCut(path)
        shortcut.Targetpath = target
        shortcut.IconLocation = icon
        shortcut.WindowStyle = 1 # 7 - Minimized, 3 - Maximized, 1 - Normal
        shortcut.save()
        os.startfile(desktop_path)
           
if __name__ == '__main__':
    app = QApplication(sys.argv)
    Window = install_window()
    Window.show()
    app.exec()
