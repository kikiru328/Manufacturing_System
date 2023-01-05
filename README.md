## Introduction

This repository contains a highly configurable two-stage-tracker that adjusts to different deployment scenarios.  
[YOLOv5](https://github.com/ultralytics/yolov5)  
[StrongSORT](https://github.com/dyhBUPT/StrongSORT)[](https://arxiv.org/pdf/2202.13514.pdf)  
[OSNet](https://github.com/KaiyangZhou/deep-person-reid)[](https://arxiv.org/abs/1905.00953),
[OCSORT](https://github.com/noahcao/OC_SORT)[](https://arxiv.org/pdf/2203.14360.pdf)  
and [ByteTrack](https://github.com/ifzhang/ByteTrack)[](https://arxiv.org/pdf/2110.06864.pdf).  
They can track any object that your Yolov5 model was trained to detect.

```
# Brief Folder Tree

📦Deepsort_yolov5
┣ 📂Archive
┃ ┣ 📂Functions
┃ ┃ ┣ 📜install_packages.py
┃ ┃ ┗ 📜make_bat_shortcut.py
┃ ┣ 📂Miniconda
┃ ┃ ┗ 📜Miniconda3-py38_22.11.1-1-Windows-x86_64.exe
┃ ┣ 📜install_gui.py
┃ ┣ 📜install_gui.spec
┃ ┣ 📜install_UI.ui
┃ ┣ 📜Manufacturing_UI.ui
┃ ┗ 📜requirements.txt
┣ 📂Functions
┃ ┣ 📂trackers
┃ ┃ ┣ 📂bytetrack
┃ ┃ ┣ 📂ocsort
┃ ┃ ┣ 📂strong_sort
┃ ┃ ┃ ┣ 📂configs
┃ ┃ ┃ ┃ ┗ 📜strong_sort.yaml
┃ ┃ ┃ ┣ 📂deep
┃ ┃ ┃ ┣ 📂results
┃ ┃ ┃ ┣ 📂sort
┃ ┃ ┃ ┣ 📂utils
┃ ┃ ┃ ┣ 📜.gitignore
┃ ┃ ┃ ┣ 📜reid_multibackend.py
┃ ┃ ┃ ┣ 📜strong_sort.py
┃ ┃ ┃ ┗ 📜**init**.py
┃ ┃ ┣ 📜multi_tracker_zoo.py
┃ ┃ ┗ 📜**init**.py
┃ ┣ 📂weights
┃ ┃ ┣ 📜best.pt
┃ ┃ ┣ 📜osnet_x0_25_msmt17.pt
┃ ┣ 📂yolov5
┃ ┣ 📜Manufacturing_function.py
┃ ┣ 📜reid_export.py
┃ ┗ 📜val.py
┣ 📜Manufacturing_gui.py
┗ 📜webcams.txt
```
