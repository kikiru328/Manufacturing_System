import argparse

import os
# limit the number of cpus used by high performance libraries
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import sys
from pathlib import Path
import numpy as np
import torch
import torch.backends.cudnn as cudnn

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # yolov5 strongsort root directory
WEIGHTS = ROOT / 'weights'


count_web_1, order_index_web_1, step_count_web_1, data_web_1 = 0, 0, 0, []
count_web_2, order_index_web_2, step_count_web_2, data_web_2 = 0, 0, 0, []
count_web_3, order_index_web_3, step_count_web_3, data_web_3 = 0, 0, 0, []





######################################################
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

import logging
from yolov5.models.common import DetectMultiBackend
from yolov5.utils.dataloaders import VID_FORMATS, LoadImages, LoadStreams
from yolov5.utils.general import (LOGGER, check_img_size, non_max_suppression, scale_coords, check_requirements, cv2,
                                  check_imshow, increment_path, strip_optimizer, colorstr, print_args, check_file)
from yolov5.utils.torch_utils import select_device, time_sync
from yolov5.utils.plots import Annotator, colors, save_one_box
from trackers.multi_tracker_zoo import create_tracker

# remove duplicated stream handler to avoid duplicated logging
logging.getLogger().removeHandler(logging.getLogger().handlers[0])

@torch.no_grad()
def run(
        source='0',
        yolo_weights=WEIGHTS / 'yolov5m.pt',  # model.pt path(s),
        reid_weights=WEIGHTS / 'osnet_x0_25_msmt17.pt',  # model.pt path,
        tracking_method='strongsort',
        imgsz=(640, 640),  # inference size (height, width)
        conf_thres=0.25,  # confidence threshold
        iou_thres=0.45,  # NMS IOU threshold
        max_det=1000,  # maximum detections per image
        device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        show_vid=False,  # show results
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
        project=ROOT / 'runs/track',  # save results to project/name
        name='exp',  # save results to project/name
        exist_ok=False,  # existing project/name ok, do not increment
        line_thickness=2,  # bounding box thickness (pixels)
        hide_labels=False,  # hide labels
        hide_conf=False,  # hide confidences
        hide_class=False,  # hide IDs
        half=False,  # use FP16 half-precision inference
        dnn=False,  # use OpenCV DNN for ONNX inference
        eval=False,  # run multi-gpu eval,
        file_path='',
        sheet_name=''
):
    
    def read_order(path, time):
        import pandas as pd
        order_data = pd.read_excel(path, sheet_name=time, header=None)
        order_data.columns = ['index', 'Name', 'Option', 'Count']
        order_data = order_data.drop('index', axis=1)
        return order_data


    order_file_path = str(file_path)
    order_sheet_name = str(sheet_name)
    #     order_file_path = './2022-08-11-제조물량_요청사항표 (수정본) copy.xlsx'
    
    # if :
    #     order_sheet_name = '요청사항표-새벽배송'
    global order_data
    order_data = read_order(order_file_path, order_sheet_name)
    
    
    
    source = str(source)
    save_img = not nosave and not source.endswith('.txt')  # save inference images
    is_file = Path(source).suffix[1:] in (VID_FORMATS)
    is_url = source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))
    webcam = source.isnumeric() or source.endswith('.txt') or (is_url and not is_file)
    if is_url and is_file:
        source = check_file(source)  # download
    
    global save_dir
    # Directories
    if not isinstance(yolo_weights, list):  # single yolo model
        exp_name = yolo_weights.stem
    elif type(yolo_weights) is list and len(yolo_weights) == 1:  # single models after --yolo_weights
        exp_name = Path(yolo_weights[0]).stem
    else:  # multiple models after --yolo_weights
        exp_name = 'ensemble'
    exp_name = name if name else exp_name + "_" + reid_weights.stem
    save_dir = increment_path(Path(project) / exp_name, exist_ok=exist_ok)  # increment run
    (save_dir / 'tracks' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Load model
    if eval:
        device = torch.device(int(device))
    else:
        device = select_device(device)
    model = DetectMultiBackend(yolo_weights, device=device, dnn=dnn, data=None, fp16=half)
    stride, names, pt = model.stride, model.names, model.pt
    imgsz = check_img_size(imgsz, s=stride)  # check image size


        
    # Dataloader
    if webcam:
        show_vid = check_imshow()
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt)
        nr_sources = len(dataset)
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt)
        nr_sources = 1
    vid_path, vid_writer, txt_path = [None] * nr_sources, [None] * nr_sources, [None] * nr_sources

    # Create as many strong sort instances as there are video sources
    tracker_list = []
    for i in range(nr_sources):
        tracker = create_tracker(tracking_method, reid_weights, device, half)
        tracker_list.append(tracker, )
        if hasattr(tracker_list[i], 'model'):
            if hasattr(tracker_list[i].model, 'warmup'):
                tracker_list[i].model.warmup()
    outputs = [None] * nr_sources

    # Run tracking
    #model.warmup(imgsz=(1 if pt else nr_sources, 3, *imgsz))  # warmup
    dt, seen = [0.0, 0.0, 0.0, 0.0], 0
    curr_frames, prev_frames = [None] * nr_sources, [None] * nr_sources
    for frame_idx, (path, im, im0s, vid_cap, s) in enumerate(dataset):

        t1 = time_sync()
        im = torch.from_numpy(im).to(device)
        im = im.half() if half else im.float()  # uint8 to fp16/32
        im /= 255.0  # 0 - 255 to 0.0 - 1.0
        if len(im.shape) == 3:
            im = im[None]  # expand for batch dim
        t2 = time_sync()
        dt[0] += t2 - t1

        # Inference
        visualize = increment_path(save_dir / Path(path[0]).stem, mkdir=True) if visualize else False
        pred = model(im, augment=augment, visualize=visualize)
        t3 = time_sync()
        dt[1] += t3 - t2

        # Apply NMS
        pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)
        dt[2] += time_sync() - t3

        # Process detections

        for i, det in enumerate(pred):  # detections per image
            seen += 1 
            # print(f'{seen:#^40}')   
            if i == 0:
                im0, save_dir, save_path, txt_file_name = webcam1(webcam, path, im, im0s, dataset, s, save_dir, source, curr_frames, line_thickness,save_crop, i, det, names, outputs, tracker_list,dt,t3,t2,tracking_method,save_txt, frame_idx, save_vid, show_vid, hide_labels, hide_conf, hide_class)
                im0, save_dir, save_path, txt_file_name = webcam1(webcam, path, im, im0s, dataset, s, save_dir, source, curr_frames, line_thickness,save_crop, i, det, names, outputs, tracker_list,dt,t3,t2,tracking_method,save_txt, frame_idx, save_vid, show_vid, hide_labels, hide_conf, hide_class)
               
            elif i == 1:
                im0, save_dir, save_path, txt_file_name = webcam2(webcam, path, im, im0s, dataset, s, save_dir, source, curr_frames, line_thickness,save_crop, i, det, names, outputs, tracker_list,dt,t3,t2,tracking_method,save_txt, frame_idx, save_vid, show_vid, hide_labels, hide_conf, hide_class)

            elif i == 2:
                im0, save_dir, save_path, txt_file_name = webcam3(webcam, path, im, im0s, dataset, s, save_dir, source, curr_frames, line_thickness,save_crop, i, det, names, outputs, tracker_list,dt,t3,t2,tracking_method,save_txt, frame_idx, save_vid, show_vid, hide_labels, hide_conf, hide_class)
            # Save results (image with detections)
            if save_vid:
      
                # global im0
                
                if vid_path[i] != save_path:  # new video
                    vid_path[i] = save_path
                    if isinstance(vid_writer[i], cv2.VideoWriter):
                        vid_writer[i].release()  # release previous video writer
                    if vid_cap:  # video
                        fps = vid_cap.get(cv2.CAP_PROP_FPS)
                        w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    else:  # stream
                        fps, w, h = 30, im0.shape[1], im0.shape[0]
                    save_path = str(Path(save_path).with_suffix('.mp4'))  # force *.mp4 suffix on results videos
                    vid_writer[i] = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                vid_writer[i].write(im0)

            prev_frames[i] = curr_frames[i]

    # Print results
    t = tuple(x / seen * 1E3 for x in dt)  # speeds per image
    LOGGER.info(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS, %.1fms {tracking_method} update per image at shape {(1, 3, *imgsz)}' % t)
    if save_txt or save_vid:
        s = f"\n{len(list(save_dir.glob('tracks/*.txt')))} tracks saved to {save_dir / 'tracks'}" if save_txt else ''
        LOGGER.info(f"Results saved to {colorstr('bold', save_dir)}{s}")
    if update:
        strip_optimizer(yolo_weights)  # update model (to fix SourceChangeWarning)


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--yolo-weights', nargs='+', type=Path, default=WEIGHTS / 'yolov5m.pt', help='model.pt path(s)')
    parser.add_argument('--reid-weights', type=Path, default=WEIGHTS / 'osnet_x0_25_msmt17.pt')
    parser.add_argument('--tracking-method', type=str, default='strongsort', help='strongsort, ocsort, bytetrack')
    parser.add_argument('--source', type=str, default='0', help='file/dir/URL/glob, 0 for webcam')  
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640], help='inference size h,w')
    parser.add_argument('--conf-thres', type=float, default=0.5, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.5, help='NMS IoU threshold')
    parser.add_argument('--max-det', type=int, default=1000, help='maximum detections per image')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--show-vid', action='store_true', help='display tracking video results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--save-crop', action='store_true', help='save cropped prediction boxes')
    parser.add_argument('--save-vid', action='store_true', help='save video tracking results')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    # class 0 is person, 1 is bycicle, 2 is car... 79 is oven
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --classes 0, or --classes 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--visualize', action='store_true', help='visualize features')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default=ROOT / 'runs/track', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--line-thickness', default=2, type=int, help='bounding box thickness (pixels)')
    parser.add_argument('--hide-labels', default=False, action='store_true', help='hide labels')
    parser.add_argument('--hide-conf', default=False, action='store_true', help='hide confidences')
    parser.add_argument('--hide-class', default=False, action='store_true', help='hide IDs')
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    parser.add_argument('--dnn', action='store_true', help='use OpenCV DNN for ONNX inference')
    parser.add_argument('--eval', action='store_true', help='run evaluation')
    parser.add_argument('--file-path', type=Path, help='order_data_excel_file_path')
    parser.add_argument('--sheet-name',type=str, help='order_data_sheet_name')
    
    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand
    print_args(vars(opt))
    return opt


def main(opt):
    check_requirements(requirements=ROOT / 'requirements.txt', exclude=('tensorboard', 'thop'))
    run(**vars(opt))
    

def webcam_start_function(webcam,path, im, im0s, dataset,s, save_dir, source, curr_frames,line_thickness,save_crop,i):
    # print(f'Process detected : {i}')
    
    if webcam:  # nr_sources >= 1
        p, im0, _ = path[i], im0s[i].copy(), dataset.count
        p = Path(p)  # to Path
        s += f'{i}: '
        txt_file_name = p.name
        save_path = str(save_dir / p.name)  # im.jpg, vid.mp4, ...
        
    else:
        p, im0, _ = path, im0s.copy(), getattr(dataset, 'frame', 0)
        p = Path(p)  # to Path
        # video file
        if source.endswith(VID_FORMATS):
            txt_file_name = p.stem
            save_path = str(save_dir / p.name)  # im.jpg, vid.mp4, ...
        # folder with imgs
        else:
            txt_file_name = p.parent.name  # get folder name containing current img
            save_path = str(save_dir / p.parent.name)  # im.jpg, vid.mp4, ...
    curr_frames[i] = im0

    txt_path = str(save_dir / 'tracks' / txt_file_name)  # im.txt
    s += '%gx%g ' % im.shape[2:]  # print string
    imc = im0.copy() if save_crop else im0  # for save_crop

    annotator = Annotator(im0, line_width=line_thickness, pil=not ascii) 
    return p, im, im0, s, txt_file_name, save_path, txt_path, imc, annotator

def common_save_functions(output, save_txt,txt_path, frame_idx, i, save_vid, save_crop, show_vid, id, cls, hide_labels, names, hide_conf, conf, hide_class, annotator, bboxes, path, imc, save_dir, p):
    if save_txt:
        # to MOT format
        bbox_left = output[0]
        bbox_top = output[1]
        bbox_w = output[2] - output[0]
        bbox_h = output[3] - output[1]
        # Write MOT compliant results to file
        with open(txt_path + '.txt', 'a') as f:
            f.write(('%g ' * 10 + '\n') % (frame_idx + 1, id, bbox_left,  # MOT format
                                        bbox_top, bbox_w, bbox_h, -1, -1, -1, i))

    if save_vid or save_crop or show_vid:  # Add bbox to image
        c = int(cls)  # integer class
        id = int(id)  # integer id
        label = None if hide_labels else (f'{id} {names[c]}' if hide_conf else \
            (f'{id} {conf:.2f}' if hide_class else f'{id} {names[c]} {conf:.2f}'))
        annotator.box_label(bboxes, label, color=colors(c, True))
        if save_crop:
            txt_file_name = txt_file_name if (isinstance(path, list) and len(path) > 1) else ''
            save_one_box(bboxes, imc, file=save_dir / 'crops' / txt_file_name / names[c] / f'{id}' / f'{p.stem}.jpg', BGR=True)
    
       #%^%^%^%^%
class Count:
    def count_1_function(det, im, s, im0, names, outputs, tracker_list, dt, i, t3,t2,tracking_method,annotator, save_txt, txt_path,frame_idx, save_vid, save_crop, show_vid, hide_labels, hide_conf, hide_class, path, imc, save_dir, p):
        w, h = im0.shape[1], im0.shape[0]
        if det is not None and len(det):
            det[:, :4] = scale_coords(im.shape[2:], det[:, :4], im0.shape).round()
            for c in det[:, -1].unique():
                n = (det[:, -1] == c).sum()  # detections per class
                s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "
            t4 = time_sync()
            outputs[i] = tracker_list[i].update(det.cpu(), im0)
            t5 = time_sync()
            dt[3] += t5 - t4
            if len(outputs[i]) > 0:
                for j, (output, conf) in enumerate(zip(outputs[i], det[:, 4])):
                    bboxes = output[0:4]
                    id = output[4]
                    cls = output[5]
                    global count_web_1, data_web_1, order_index_web_1, step_count_web_1
                    center_coordinates = (
                        int(bboxes[0] + (bboxes[2]-bboxes[0])/2), int(bboxes[1] + (bboxes[3] - bboxes[1])/2))
                    
                    if (int(bboxes[0]+(bboxes[2] - bboxes[0])/2) < (int(w/2))) and (id not in data_web_1):

                        # like sensor
                        im0 = cv2.rectangle(im0, (0,0), (w,h), (0,0,255), -1)
                        
                        count_web_1 += 1
                        data_web_1.append(id)
                        order_data_count = order_data['Count']
                        step_count_web_1 += 1
                        
                        if step_count_web_1 >= int(order_data_count[order_index_web_1]):
                            order_index_web_1+= 1
                            step_count_web_1 = 0 
                            
                    common_save_functions(output, save_txt, txt_path, frame_idx, i, save_vid, save_crop, show_vid, id, cls,hide_labels, names, hide_conf, conf, hide_class, annotator, bboxes, path, imc, save_dir, p)
                    
            LOGGER.info(f'{s}Done. yolo:({t3 - t2:.3f}s), {tracking_method}:({t5 - t4:.3f}s)')
        else:
            #strongsort_list[i].increment_ages()
            LOGGER.info('No detections')    
        im0 = annotator.result()
        return im0, count_web_1, order_index_web_1, step_count_web_1
    
    
    def count_2_function(det, im, s, im0, names, outputs, tracker_list, dt, i, t3,t2,tracking_method,annotator, save_txt, txt_path,frame_idx, save_vid, save_crop, show_vid, hide_labels, hide_conf, hide_class, path, imc, save_dir, p):
        w, h = im0.shape[1], im0.shape[0]
        if det is not None and len(det):
            det[:, :4] = scale_coords(im.shape[2:], det[:, :4], im0.shape).round()
            for c in det[:, -1].unique():
                n = (det[:, -1] == c).sum()  # detections per class
                s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "
            t4 = time_sync()
            outputs[i] = tracker_list[i].update(det.cpu(), im0)
            t5 = time_sync()
            dt[3] += t5 - t4
            if len(outputs[i]) > 0:
                for j, (output, conf) in enumerate(zip(outputs[i], det[:, 4])):
                    bboxes = output[0:4]
                    id = output[4]
                    cls = output[5]
                    global count_web_2, data_web_2, order_index_web_2, step_count_web_2
                    center_coordinates = (
                        int(bboxes[0] + (bboxes[2]-bboxes[0])/2), int(bboxes[1] + (bboxes[3] - bboxes[1])/2))
                    
                    if (int(bboxes[0]+(bboxes[2] - bboxes[0])/2) < (int(w/2))) and (id not in data_web_2):
                        im0 = cv2.rectangle(im0, (0,0), (w,h), (0,0,255), -1)
                        count_web_2 += 1
                        data_web_2.append(id)
                        order_data_count = order_data['Count']
                        step_count_web_2 += 1 
                        if step_count_web_2 >= int(order_data_count[order_index_web_2]):
                            order_index_web_2+= 1
                            step_count_web_2 = 0
                            
                    common_save_functions(output, save_txt,txt_path, frame_idx, i, save_vid, save_crop, show_vid, id, cls,hide_labels, names, hide_conf, conf, hide_class, annotator, bboxes, path, imc, save_dir, p)
                    
            LOGGER.info(f'{s}Done. yolo:({t3 - t2:.3f}s), {tracking_method}:({t5 - t4:.3f}s)')
        else:
            #strongsort_list[i].increment_ages()
            LOGGER.info('No detections')    
        im0 = annotator.result()
        return im0, count_web_2, order_index_web_2, step_count_web_2
    
def count_3_function(det, im, s, im0, names, outputs, tracker_list, dt, i, t3,t2,tracking_method,annotator, save_txt, txt_path,frame_idx, save_vid, save_crop, show_vid, hide_labels, hide_conf, hide_class, path, imc, save_dir, p):
        w, h = im0.shape[1], im0.shape[0]
        if det is not None and len(det):
            det[:, :4] = scale_coords(im.shape[2:], det[:, :4], im0.shape).round()
            for c in det[:, -1].unique():
                n = (det[:, -1] == c).sum()  # detections per class
                s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "
            t4 = time_sync()
            outputs[i] = tracker_list[i].update(det.cpu(), im0)
            t5 = time_sync()
            dt[3] += t5 - t4
            if len(outputs[i]) > 0:
                for j, (output, conf) in enumerate(zip(outputs[i], det[:, 4])):
                    bboxes = output[0:4]
                    id = output[4]
                    cls = output[5]
                    global count_web_3, data_web_3, order_index_web_3, step_count_web_3
                    center_coordinates = (
                        int(bboxes[0] + (bboxes[2]-bboxes[0])/2), int(bboxes[1] + (bboxes[3] - bboxes[1])/2))
                    
                    if (int(bboxes[0]+(bboxes[2] - bboxes[0])/2) < (int(w/2))) and (id not in data_web_3):
                        im0 = cv2.rectangle(im0, (0,0), (w,h), (0,0,255), -1)
                        count_web_3 += 1
                        data_web_3.append(id)
                        order_data_count = order_data['Count']
                        step_count_web_3 += 1 
                        if step_count_web_3 >= int(order_data_count[order_index_web_2]):
                            order_index_web_3+= 1
                            step_count_web_3 = 0
                            
                    common_save_functions(output, save_txt,txt_path, frame_idx, i, save_vid, save_crop, show_vid, id, cls,hide_labels, names, hide_conf, conf, hide_class, annotator, bboxes, path, imc, save_dir, p)
                    
            LOGGER.info(f'{s}Done. yolo:({t3 - t2:.3f}s), {tracking_method}:({t5 - t4:.3f}s)')
        else:
            #strongsort_list[i].increment_ages()
            LOGGER.info('No detections')    
        im0 = annotator.result()
        return im0, count_web_3, order_index_web_3, step_count_web_3


def draw_text(img, org, text, color, font_size):
    from PIL import Image, ImageFont, ImageDraw
    import numpy as np
    text_ = Image.fromarray(img)
    draw = ImageDraw.Draw(text_)
    try:
        font = ImageFont.truetype("C:/Windows/Fonts/batang.ttc", font_size)
    except:
        font = ImageFont.truetype("/Volumes/Macintosh HD/System/Library/Fonts/AppleSDGothicNeo.ttc", font_size)
    draw.text(org, text, font = font, color = color)
    return np.array(text_)
     

def basic_draw_function(im0, count, order_index, step_count):
    w, h = im0.shape[1], im0.shape[0]
    from PIL import Image
    from PIL import ImageFont
    from PIL import ImageDraw
    imb = np.zeros(im0.shape, np.uint8)
    color = (0,255,0)
    start_point = (int(w/2), 0)
    end_point = (int(w/2), h)
    cv2.line(im0, start_point, end_point, color, thickness=2)
    background = Image.fromarray(imb)
    draw = ImageDraw.Draw(background)
    total_count_text_org = (200,200)
    total_count_text = f"총 진행 개수 : {str(count)}"
    order_count_text_org = (200,300)
    order_count_text = f"현재 옵션 : {order_data['Option'][order_index]}            ({step_count} / {order_data['Count'][order_index]})"
    try:
        next_option_text_org = (200,400)
        next_option_text = f"다음 옵션 : {order_data['Option'][order_index+1]}"
    except:
        next_option_text = f"다음 옵션 : 현재가 마지막 옵션입니다."
    alpha = 0.3
    try:
        font = ImageFont.truetype("C:/Windows/Fonts/batang.ttc", 11)
    except:
        font = ImageFont.truetype("/Volumes/Macintosh HD/System/Library/Fonts/AppleSDGothicNeo.ttc", 11)
    draw.text(total_count_text_org, total_count_text, font = font, fill = (0, 255, 0))
    draw.text(order_count_text_org, order_count_text, font=font, fill=(0, 255, 0))
    draw.text(next_option_text_org, next_option_text, font=font, fill=(0,255,0))
    background_with_text = np.array(background)
    add_image = cv2.addWeighted(im0, alpha, background_with_text, (1-alpha), 0)
    return add_image

def toping_draw_function(im0, count, order_index, step_count): 
    w, h = im0.shape[1], im0.shape[0]
    
    from PIL import Image
    from PIL import ImageFont
    from PIL import ImageDraw
    
    # Background
    imb = np.zeros(im0.shape, np.uint8)
    # print(w,h)
    # Resize
    resize_im0 = imb.copy()
    resizeas = (w, int(h*0.8))
    # resizeas = (100, 100)
    im0 = cv2.resize(im0, dsize=resizeas,interpolation= cv2.INTER_LINEAR)
    resize_im0[:int(h*0.8),:,:] = im0
    # resize_im0[h-110:h-10, 50:150, :] = im0
    
    # count line
    color = (255,0,0)
    start_point = (int(w/2), 0)
    end_point = (int(w/2), int(h*0.77))
    cv2.line(resize_im0, start_point, end_point, color, thickness=3)
    
    # CHECK BOX
    option = order_data['Option'][order_index]
    
    # BOX
    overlay = resize_im0.copy()
    box_alpha = 0.2
    
    #basic check box
    if option != str:
        option = str(option)
    
    if '콩x' in option:
        box = cv2.rectangle(overlay, (0,0),(int(w/2),int(h*0.8)), (0,0,255), -1)
    else:
        box = cv2.rectangle(overlay, (0,0),(int(w/2),int(h*0.8)), (0,100,0), -1)
            
    if '당근x' in option:
        box = cv2.rectangle(overlay, (int(w/2),0),(w,int(h*0.8)), (0,0,255), -1)
    else:
        box = cv2.rectangle(overlay, (int(w/2),0),(w,int(h*0.8)), (0,100,0), -1)    
        
        
    final_check = cv2.addWeighted(resize_im0, (1-box_alpha), box, box_alpha, 0)
    
    
    if '콩x' in option:
        final_check = draw_text(final_check, (int(w*0.2), int(h*0.35)), "콩 X", (0,0,0), 50)
    else:
        final_check = draw_text(final_check, (int(w*0.2), int(h*0.35)), "콩 O", (0,0,0), 50)
    
    if '당근x' in option:
        final_check = draw_text(final_check, (int(w*0.65), int(h*0.35)), "당근 X", (0,0,0), 50)
    else:
        final_check = draw_text(final_check, (int(w*0.65), int(h*0.35)), "당근 O", (0,0,0), 50)
        
                
    # devide_line
    sp = (int(w/2),0)
    ep = (int(w/2),int(h*0.8))
    cv2.line(final_check, sp, ep, (0,0,0), thickness = 5 )
    
    # Draw 
    # background = Image.fromarray(imb)
    # draw = ImageDraw.Draw(background)
    text_ = Image.fromarray(final_check)
    draw = ImageDraw.Draw(text_)
    
    # Text
    # total_count_text_org = (int(w*0.45), int(h*0.83))
    total_count_text_org = (int(w*0.05), int(h*0.83))
    total_count_text = f"총 진행 개수 : {str(count)}"
    # order_count_text_org = (int(w*0.45), int(h*0.88))
    order_count_text_org = (int(w*0.05), int(h*0.88))
    order_count_text = f"현재 옵션 : {order_data['Option'][order_index]}            ({step_count} / {order_data['Count'][order_index]})"
    try:
        # next_option_text_org = (int(w*0.45), int(h*0.93))
        next_option_text_org = (int(w*0.05), int(h*0.93))
        next_option_text = f"다음 옵션 : {order_data['Option'][order_index+1]}"  
    except:
        next_option_text = f"다음 옵션 : 현재가 마지막 옵션입니다."
    try:
        font = ImageFont.truetype("C:/Windows/Fonts/batang.ttc", 20)
    except:
        font = ImageFont.truetype("/Volumes/Macintosh HD/System/Library/Fonts/AppleSDGothicNeo.ttc", 20)
    draw.text(total_count_text_org, total_count_text, font = font, fill = (0,255, 255))
    draw.text(order_count_text_org, order_count_text, font=font, fill=(0, 255, 255))
    draw.text(next_option_text_org, next_option_text, font=font, fill=(0,255,255))

    add_image = np.array(text_)
    return add_image

def A_draw_function(im0, count, order_index, step_count): 
    w, h = im0.shape[1], im0.shape[0]
    
    from PIL import Image
    from PIL import ImageFont
    from PIL import ImageDraw
    
    # Background
    imb = np.zeros(im0.shape, np.uint8)
    
    # Resize
    resize_im0 = imb.copy()
    resizeas = (w, int(h*0.8))
    im0 = cv2.resize(im0, dsize=resizeas)
    resize_im0[:int(h*0.8),:,:] = im0
    
    # count line
    color = (0,0,0)
    start_point = (int(w/2), 0)
    end_point = (int(w/2), int(h*0.79))
    cv2.line(resize_im0, start_point, end_point, color, thickness=3)
    
    # CHECK BOX
    option = order_data['Option'][order_index]
    
    # BOX
    overlay = resize_im0.copy()
    box_alpha = 0.2
    
    #basic check box
    if option != str:
        option = str(option)
    
    
    # box = cv2.rectangle(overlay, (0,0), (0,int(h*0.8)), (0,0,255), -1)
    box = cv2.rectangle(overlay, (0,0), (w,int(h*0.8)), (0,0,0), -1)
    final_check = cv2.addWeighted(resize_im0, (1-box_alpha), box, box_alpha, 0)
    final_check = draw_text(final_check, (int(w*0.3), int(h*0.35)), option, (0,0,0), 50)
          
    # devide_line
    # sp = (int(w/2),0)
    # ep = (int(w/2),int(h*0.8))
    # cv2.line(final_check, sp, ep, (0,0,0), thickness = 5 )
    
    # Draw 
    # background = Image.fromarray(imb)
    # draw = ImageDraw.Draw(background)
    text_ = Image.fromarray(final_check)
    draw = ImageDraw.Draw(text_)
    
    # Text
    total_count_text_org = (int(w*0.45), int(h*0.83))
    total_count_text = f"총 진행 개수 : {str(count)}"
    order_count_text_org = (int(w*0.45), int(h*0.88))
    order_count_text = f"현재 옵션 : {order_data['Option'][order_index]}            ({step_count} / {order_data['Count'][order_index]})"
    try:
        next_option_text_org = (int(w*0.45), int(h*0.93))
        next_option_text = f"다음 옵션 : {order_data['Option'][order_index+1]}"  
    except:
        next_option_text = f"다음 옵션 : 현재가 마지막 옵션입니다."
    try:
        font = ImageFont.truetype("C:/Windows/Fonts/batang.ttc", 20)
    except:
        font = ImageFont.truetype("/Volumes/Macintosh HD/System/Library/Fonts/AppleSDGothicNeo.ttc", 20)
    draw.text(total_count_text_org, total_count_text, font = font, fill = (0,255, 255))
    draw.text(order_count_text_org, order_count_text, font=font, fill=(0, 255, 255))
    draw.text(next_option_text_org, next_option_text, font=font, fill=(0,255,255))

    add_image = np.array(text_)
    return add_image




def finish_img(im0):
    finish_blank = np.zeros(im0.shape, np.uint8)
    w, h = finish_blank.shape[1], finish_blank.shape[0]
    
    from PIL import Image
    from PIL import ImageFont
    from PIL import ImageDraw
    text_ = Image.fromarray(finish_blank)
    draw = ImageDraw.Draw(text_)
    try:
        font = ImageFont.truetype("C:/Windows/Fonts/batang.ttc", 20)
    except:
        font = ImageFont.truetype("/Volumes/Macintosh HD/System/Library/Fonts/AppleSDGothicNeo.ttc", 20)
        
    draw.text((int(w/3), int(h/2)), "제조가 종료되었습니다.\n고생하셨습니다.", font=font, fill=(255,255,255))
    add_image = np.array(text_)
    return add_image



def screen_show(show_vid, i, add_image):
    if show_vid:
        import screeninfo
        screen_id = i
        screen = screeninfo.get_monitors()[screen_id]               
        screen_width, screen_height = screen.width, screen.height
        
        add_image = cv2.resize(add_image, (screen_width, screen_height))
        
        add_image[0,0] = 0
        add_image[screen_height-2, 0] = 0
        add_image[0, screen_width-2] = 0
        add_image[screen_height-2, screen_width-2] = 0
        window_name = str(i)
        # window_name = 'A'
        # cv2.namedWindow(window_name, cv2.WND_PROP_FULLSCREEN)
        # cv2.moveWindow(window_name, screen.x -1, screen.y-1)
        # cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
            
        cv2.imshow(window_name, add_image)
        cv2.waitKey(1)

        
def webcam1(webcam, path, im, im0s, dataset, s, save_dir, source, curr_frames, line_thickness,save_crop, i, det, names, outputs, tracker_list,dt,t3,t2,tracking_method,save_txt, frame_idx, save_vid, show_vid, hide_labels, hide_conf, hide_class):
    try:
        p, im, im0, s, txt_file_name, save_path, txt_path, imc, annotator = webcam_start_function(webcam,path, im, im0s, dataset,s, save_dir, source, curr_frames,line_thickness,save_crop,i)
        
        im0, count_web_1, order_index_web_1, step_count_web_1 = Count.count_1_function(det, im, s, im0, names, outputs, tracker_list, dt, i, t3,t2,tracking_method,annotator, save_txt, txt_path,frame_idx, save_vid, save_crop, show_vid, hide_labels, hide_conf, hide_class, path, imc, save_dir, p)
        
        add_image = toping_draw_function(im0 = im0, count = count_web_1, order_index = order_index_web_1, step_count = step_count_web_1)
    except Exception as e:
        # print(f'{str(e):#^20}')
        add_image = finish_img(im0 = im0)
    
    screen_show(show_vid, i, add_image)
    return im0, save_dir, save_path, txt_file_name
    
        

def webcam2(webcam, path, im, im0s, dataset, s, save_dir, source, curr_frames, line_thickness,save_crop, i, det, names, outputs, tracker_list,dt,t3,t2,tracking_method,save_txt, frame_idx, save_vid, show_vid, hide_labels, hide_conf, hide_class):
    try:
        p, im, im0, s, txt_file_name, save_path, txt_path, imc, annotator = webcam_start_function(webcam,path, im, im0s, dataset,s, save_dir, source, curr_frames,line_thickness,save_crop,i)
        
        im0, count_web_2, order_index_web_2, step_count_web_2 = Count.count_2_function(det, im, s, im0, names, outputs, tracker_list, dt, i, t3,t2,tracking_method,annotator, save_txt, txt_path,frame_idx, save_vid, save_crop, show_vid, hide_labels, hide_conf, hide_class, path, imc, save_dir, p)
        
        add_image = A_draw_function(im0 = im0, count = count_web_2, order_index = order_index_web_2, step_count = step_count_web_2)
    except Exception as e:
        # print(f'{str(e):#^20}')
        add_image = finish_img(im0 = im0)
        
    screen_show(show_vid, i, add_image)
    return im0, save_dir, save_path, txt_file_name

def webcam3(webcam, path, im, im0s, dataset, s, save_dir, source, curr_frames, line_thickness,save_crop, i, det, names, outputs, tracker_list,dt,t3,t2,tracking_method,save_txt, frame_idx, save_vid, show_vid, hide_labels, hide_conf, hide_class):
    try:
        p, im, im0, s, txt_file_name, save_path, txt_path, imc, annotator = webcam_start_function(webcam,path, im, im0s, dataset,s, save_dir, source, curr_frames,line_thickness,save_crop,i)
        
        im0, count_web_3, order_index_web_3, step_count_web_3 = Count.count_3_function(det, im, s, im0, names, outputs, tracker_list, dt, i, t3,t2,tracking_method,annotator, save_txt, txt_path,frame_idx, save_vid, save_crop, show_vid, hide_labels, hide_conf, hide_class, path, imc, save_dir, p)
        
        add_image = A_draw_function(im0 = im0, count = count_web_3, order_index = order_index_web_3, step_count = step_count_web_3)
    except Exception as e:
        # print(f'{str(e):#^20}')
        add_image = finish_img(im0 = im0)
        
    screen_show(show_vid, i, add_image)
    return im0, save_dir, save_path, txt_file_name
    
    




if __name__ == "__main__":
    opt = parse_opt()
    main(opt)