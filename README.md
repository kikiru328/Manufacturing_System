#### Reference
This repository contains a highly configurable two-stage-tracker that adjusts to different deployment scenarios.  

#### [YOLOv5](https://github.com/ultralytics/yolov5) for Detection  [StrongSORT](https://github.com/dyhBUPT/StrongSORT)[](https://arxiv.org/pdf/2202.13514.pdf) for Tracking


# Introduction
컨베이어 벨트 시스템은 제조 공정에서 순서대로 제조를 가능하게 하는 장점이 있다.  
하지만 사람이 제조할 시 불량률이 기계보다 높게 나온다는 단점이 있다. 또한 옵션이 많을 경우에는 불량률이 더욱더 증가한다.  
  
따라서 제조시 옵션을 상시 제공하여 불량률을 줄일 수 있는 시스템을 개발해보았다.  

객체 추적은 `yolov5-s` 모델을 사용하였고, 추적을 추적을 위해 `Deepsort` 알고리즘을 사용하였다.  
이 프로그램은 추적 뿐 아니라 개수를 세야하기 떄문에 객체 중앙점을 기점으로 객체의 중앙점이 넘어설 때, count += 1 이 되며  해당 옵션을 제공하게 된다.  

해당 옵션은 `screeninfo`를 이용해서 모니터에 제공한다.  

GUI는 pyqt5로 개발하여 사용자의 편리성을 높였다.  


The conveyor belt system has an advantage of enabling manufacturing in order in a manufacturing process.
However, there is a disadvantage that the defect rate is higher than that of machines when manufactured by humans. In addition, if there are many options, the defect rate increases even more.  

Therefore, I have developed a system that can reduce the defect rate by always providing options during manufacturing.  

The 'yolov5-s' model was used for object tracking, and the 'Deepsort' algorithm was used for tracking.
Because this program requires not only tracking but also counting, when the center point of the object exceeds the center point of the object, count + = 1 is given and the corresponding option is provided.  

This option is provided to the monitor using `'screeninfo'`.  
 
GUI was developed as`pyqt5` to increase user convenience.  
```
# Brief Folder Tree

📦Deepsort_yolov5
┣ 📂Archive
┃ ┣ 📂Functions
┃ ┃ ┣ 📜install_packages.py      ━━━ # install requirements
┃ ┃ ┗ 📜make_bat_shortcut.py
┃ ┣ 📂Miniconda                  ━━━ # For distribute environments
┃ ┣ 📜install_gui.py             ━━━ # install program gui
┃ ┣ 📜install_UI.ui
┃ ┣ 📜Manufacturing_UI.ui
┃ ┗ 📜requirements.txt           ━━━ # For distribute environments
┣ 📂Functions
┃ ┣ 📂trackers
┃ ┃ ┣ 📂bytetrack
┃ ┃ ┣ 📂ocsort
┃ ┃ ┣ 📂strong_sort
┃ ┃ ┃ ┣ 📂configs
┃ ┃ ┃ ┃ ┗ 📜strong_sort.yaml    ━┓
┃ ┃ ┃ ┣ 📂sort                   ┃
┃ ┃ ┃ ┣ 📜reid_multibackend.py   ┃
┃ ┃ ┃ ┣ 📜strong_sort.py         ┃
┃ ┃ ┃ ┗ 📜__init__.py            ┃
┃ ┃ ┣ 📜multi_tracker_zoo.py    ━┻━ # Need to match the path "cfg.merge_from_file(Strong_sort.yaml Path)")
┃ ┃ ┗ 📜__init__.py
┃ ┣ 📂weights                   ━━━ # weigths for yolov5 model and strong_sort
┃ ┃ ┣ 📜best.pt
┃ ┃ ┣ 📜osnet_x0_25_msmt17.pt
┃ ┣ 📂yolov5
┃ ┃ ┣ 📂.github
┃ ┃ ┣ 📂classify
┃ ┃ ┣ 📂data
┃ ┃ ┣ 📂models
┃ ┃ ┣ 📂utils
┃ ┃ ┃ ┣ 📜plots.py             ━━━ # Find center point
┃ ┣ 📜Manufacturing_function.py
┃ ┣ 📜reid_export.py
┃ ┗ 📜val.py
┣ 📜Manufacturing_gui.py       ━━━ # GUI for using program (pyqt5) 
┗ 📜webcams.txt                ━━━ # Input Videos

```

# Find Center point
Yolov5 - Utils - plots 내부에 아래와 같이 추가한다.  
<details>
<summary><font size='2'>Plots_code</font></summary>
<div markdown='1'>

~~~python
    def box_label(self, box, label='', color=(128, 128, 128), txt_color=(255, 255, 255)):
        # Add one xyxy box to image with label
        if self.pil or not is_ascii(label):
            self.draw.rectangle(box, width=self.lw, outline=color)  # box
            if label:
                w, h = self.font.getsize(label)  # text width, height
                outside = box[1] - h >= 0  # label fits outside box
                self.draw.rectangle(
                    (box[0], box[1] - h if outside else box[1], box[0] + w + 1,
                     box[1] + 1 if outside else box[1] + h + 1),
                    fill=color,
                )
                # self.draw.text((box[0], box[1]), label, fill=txt_color, font=self.font, anchor='ls')  # for PIL>8.0
                self.draw.text((box[0], box[1] - h if outside else box[1]), label, fill=txt_color, font=self.font)
        else:  # cv2
            p1, p2 = (int(box[0]), int(box[1])), (int(box[2]), int(box[3]))
            center_coordinates = (int(box[0] + (box[2]-box[0])/2), int(box[1] + (box[3] - box[1])/2))
            cv2.rectangle(self.im, p1, p2, color, thickness=self.lw, lineType=cv2.LINE_AA)
            cv2.circle(self.im, center_coordinates, radius=3, color=color, thickness=3)
            if label:
                tf = max(self.lw - 1, 1)  # font thickness
                w, h = cv2.getTextSize(label, 0, fontScale=self.lw / 3, thickness=tf)[0]  # text width, height
                outside = p1[1] - h >= 3
                p2 = p1[0] + w, p1[1] - h - 3 if outside else p1[1] + h + 3
                cv2.rectangle(self.im, p1, p2, color, -1, cv2.LINE_AA)  # filled
                cv2.putText(self.im,
                            label, (p1[0], p1[1] - 2 if outside else p1[1] + h + 2),
                            0,
                            self.lw / 3,
                            txt_color,
                            thickness=tf,
                            lineType=cv2.LINE_AA
~~~                            
</div>
</details>

![Box_Label in Yolov5](https://user-images.githubusercontent.com/60537388/210808158-dd82fed1-82de-49de-8aee-bc25039e19ba.png){: width='20%' height='20%' .align-center}



