# YOLOv5 🚀 by Ultralytics, AGPL-3.0 license
"""
Run YOLOv5 detection inference on images, videos, directories, globs, YouTube, webcam, streams, etc.

Usage - sources:
    $ python detect.py --weights yolov5s.pt --source 0                               # webcam
                                                     img.jpg                         # image
                                                     vid.mp4                         # video
                                                     screen                          # screenshot
                                                     path/                           # directory
                                                     list.txt                        # list of images
                                                     list.streams                    # list of streams
                                                     'path/*.jpg'                    # glob
                                                     'https://youtu.be/LNwODJXcvt4'  # YouTube
                                                     'rtsp://example.com/media.mp4'  # RTSP, RTMP, HTTP stream

Usage - formats:
    $ python detect.py --weights yolov5s.pt                 # PyTorch
                                 yolov5s.torchscript        # TorchScript
                                 yolov5s.onnx               # ONNX Runtime or OpenCV DNN with --dnn
                                 yolov5s_openvino_model     # OpenVINO
                                 yolov5s.engine             # TensorRT
                                 yolov5s.mlmodel            # CoreML (macOS-only)
                                 yolov5s_saved_model        # TensorFlow SavedModel
                                 yolov5s.pb                 # TensorFlow GraphDef
                                 yolov5s.tflite             # TensorFlow Lite
                                 yolov5s_edgetpu.tflite     # TensorFlow Edge TPU
                                 yolov5s_paddle_model       # PaddlePaddle
"""
import platform
import pathlib
plt = platform.system()
if plt != 'Windows':
  pathlib.WindowsPath = pathlib.PosixPath
import argparse
import csv
from MeterClass import *
import os
import platform
import sys
from pathlib import Path

import torch

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from ultralytics.utils.plotting import Annotator, colors,save_one_box

from models.common import DetectMultiBackend
from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadScreenshots, LoadStreams
from utils.general import (LOGGER, Profile, check_file, check_img_size, check_imshow, check_requirements, colorstr, cv2,
                           increment_path, non_max_suppression, print_args, scale_boxes, strip_optimizer, xyxy2xywh)
from utils.torch_utils import select_device, smart_inference_mode


@smart_inference_mode()
def run(
    angle = 266,
    range = 100,
    weights=ROOT / 'best.pt',  # model path or triton URL
    source=ROOT / 'data/images',  # file/dir/URL/glob/screen/0(webcam)
    data=ROOT / 'data.yaml',  # dataset.yaml path
    imgsz=(640, 640),  # inference size (height, width)
    conf_thres=0.25,  # confidence threshold
    iou_thres=0.45,  # NMS IOU threshold
    max_det=1000,  # maximum detections per image
    device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
    view_img=False,  # show results
    save_txt=False,  # save results to *.txt
    save_csv=False,  # save results in CSV format
    save_conf=False,  # save confidences in --save-txt labels
    save_crop=False,  # save cropped prediction boxes
    nosave=False,  # do not save images/videos
    classes=None,  # filter by class: --class 0, or --class 0 2 3
    agnostic_nms=False,  # class-agnostic NMS
    augment=False,  # augmented inference
    visualize=False,  # visualize features
    update=False,  # update all models
    project=ROOT / 'runs/detect',  # save results to project/name
    name='exp',  # save results to project/name
    exist_ok=False,  # existing project/name ok, do not increment
    line_thickness=3,  # bounding box thickness (pixels)
    hide_labels=False,  # hide labels
    hide_conf=False,  # hide confidences
    half=False,  # use FP16 half-precision inference
    dnn=False,  # use OpenCV DNN for ONNX inference
    vid_stride=1,  # video frame-rate stride
    non_none_count=0,
    sum_count=0,
    sum_max=0,
    sum_min=0,
    readValue=0,
    readValue_old=0,
    stable_flag=None,
    avertime=0,
    sumtime = 0,
    avertime_data = 0.01,
    FPS = 0,
    heights=[],
    widths=[],
    unstable=0,
    readValue_lst=[0] * 20,
    index = 0,
    filter_Value = 0,
    ):
    source = str(source)
    save_img = not nosave and not source.endswith('.txt')  # save inference images
    is_file = Path(source).suffix[1:] in (IMG_FORMATS + VID_FORMATS)
    is_url = source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))
    webcam = source.isnumeric() or source.endswith('.streams') or (is_url and not is_file)
    screenshot = source.lower().startswith('screen')
    if is_url and is_file:
        source = check_file(source)  # download

    # Directories
    save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Load model
    device = select_device(device)
    model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data, fp16=half)
    stride, names, pt = model.stride, model.names, model.pt
    imgsz = check_img_size(imgsz, s=stride)  # check image size

    # Dataloader
    bs = 1  # batch_size
    if webcam:
        view_img = check_imshow(warn=True)
        dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)
        bs = len(dataset)
    elif screenshot:
        dataset = LoadScreenshots(source, img_size=imgsz, stride=stride, auto=pt)
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)
    vid_path, vid_writer = [None] * bs, [None] * bs

    # Run inference
    model.warmup(imgsz=(1 if pt or model.triton else bs, 3, *imgsz))  # warmup
    seen, windows, dt = 0, [], (Profile(), Profile(), Profile(),Profile(),Profile())
    for path, im, im0s, vid_cap, s in dataset:
        with dt[3]:
            im = torch.from_numpy(im).to(model.device)
            im = im.half() if model.fp16 else im.float()  # uint8 to fp16/32
            im /= 255  # 0 - 255 to 0.0 - 1.0
            if len(im.shape) == 3:
                im = im[None]  # expand for batch dim

            if stable_flag is None:
            # Inference
                visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if visualize else False
                pred = model(im, augment=augment, visualize=visualize)
            # NMS
                pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)

            # Process predictions
            with dt[2]:
                for i, det in enumerate(pred):  # per image
                    seen += 1
                    if webcam:  # batch_size >= 1
                        p, im0, frame = path[i], im0s[i].copy(), dataset.count
                        s += f'{i}: '
                    else:
                        p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)

                    p = Path(p)  # to Path
                    save_path = str(save_dir / p.name)  # im.jpg
                    s += '%gx%g ' % im.shape[2:]  # print string
                    imc = im0.copy() if save_crop else im0  # for save_crop
                    annotator = Annotator(im0, line_width=line_thickness, example=str(names))
                    if len(det):
                        # Rescale boxes from img_size to im0 size
                        det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()

                        # Print results
                        for c in det[:, 5].unique():
                            n = (det[:, 5] == c).sum()  # detections per class
                            s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                        # Write results
                        for *xyxy, conf, cls in reversed(det):
                            c = int(cls)  # integer class
                            '''save_one_box时间:5.8ms,MeterDetection时间:0.1ms'''
                            if stable_flag is not None:
                                A = MeterDetection(save_one_box(xyxy, imc, save_dir / 'crops' / names[c] / f'{p.stem}.jpg', BGR=True,save=True))
                                if readValue_old == readValue and readValue is None:
                                     unstable += 1
                                if unstable >= FPS*1.5:
                                    stable_flag = None
                                    unstable = 0
                                readValue_old = readValue
                                '''140ms'''
                                with dt[0]:
                                    readValue = A.Readvalue(range, angle)
                                '''≈0ms'''
                                if readValue != 0 and readValue is not None:
                                    if readValue > range and readValue_old is not None:
                                        readValue = readValue_old
                                    readValue = readValue + 3.5
                                    readValue_lst[index] = readValue
                                    index += 1
                                    if index >= 20:
                                        index = 0
                                    filter_Value = sum(readValue_lst) / len(readValue_lst)
                                    print(f"滤波后读数值{filter_Value:.2f}",f"实际读数值{readValue:.2f},")
                                    non_none_count += 1
                                    sum_max = max(readValue_lst)
                                    sum_min = min(readValue_lst)
                                    print(f"近十次读数最大值:{sum_max:.2f}",f"读数最小值:{sum_min:.2f}")
                                sum_count += 1
                                miss_detection_percentage = (non_none_count) / sum_count * 100
                                print("程序运行次数:", sum_count,"读数检测率: {:.2f}%".format(miss_detection_percentage))
                            '''0.5ms'''
                            if save_img or save_crop or view_img:  # Add bbox to image
                                c = int(cls)  # integer class
                                if readValue is None:
                                    label = None if hide_labels else (names[c] if hide_conf else f'NoneRead {conf:.2f}')
                                    annotator.box_label(xyxy, label, color=colors(c, True))
                                else:
                                    label = None if hide_labels else (names[c] if hide_conf else f'{filter_Value/1000:.4f}{"Mpa"}')
                                    annotator.box_label(xyxy, label, color=colors(c, True))
                            '''≈0ms'''
                            if stable_flag is None:
                                x1,y1,x2,y2 = xyxy
                                heights.append(int(abs(y1.item() - y2.item())))
                                widths.append(int(abs(x1.item() - x2.item())))
                                print(len(heights))
                                if len(heights) > 10 or len(widths) > 10:
                                    min_height = min(heights)  # Calculate the minimum value and its index
                                    min_width = min(widths)
                                    min_height_index = heights.index(min_height)
                                    min_width_index = widths.index(min_width)
                                    heights.pop(min_height_index)  # Remove the minimum values from the list
                                    widths.pop(min_width_index)
                                    if len(heights) > 0:
                                        height_diff_aver = sum([abs(h - min_height) for h in heights]) / len(heights)  # Calculate the sum of differences
                                    else:
                                        height_diff_aver = 0

                                    if len(widths) > 0:
                                        width_diff_aver = sum([abs(w - min_width) for w in widths]) / len(widths)
                                    else:
                                        width_diff_aver = 0

                                    if height_diff_aver < 2 or width_diff_aver < 2:
                                        stable_flag = True

                                    heights = []
                                    widths = []

                    cv2.putText(im0, str(f"FPS:{FPS:.2f}"), (0, 40), cv2.FONT_HERSHEY_SIMPLEX, 1,(0, 255, 0), 2)
                    if stable_flag is None:
                        cv2.putText(im0, str(f"initial:{len(heights) * 10}%,{len(widths) * 10}%"), (0, 80), cv2.FONT_HERSHEY_SIMPLEX, 1,(0, 255, 0), 2)
                    else:
                        cv2.putText(im0, str(f"initial:OK"), (0, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0),2)

                    # Stream results
                    im0 = annotator.result()
                    '''9.5ms'''
                    if view_img:
                        if platform.system() == 'Linux' and p not in windows:
                            windows.append(p)
                            cv2.namedWindow(str(p), cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)  # allow window resize (Linux)
                            cv2.resizeWindow(str(p), im0.shape[1], im0.shape[0])
                        cv2.imshow(str(p), im0)
                        cv2.waitKey(1)  # 1 millisecond
                    # Save results (image with detections)
                    if save_img:
                        if dataset.mode == 'image':
                            if not os.path.exists(save_dir / f'{p.stem}'):  # 是否存在这个文件夹
                                os.makedirs(save_dir / f'{p.stem}')  # 如果没有这个文件夹，那就创建一个
                            cv2.imwrite(save_dir / f'{p.stem}' / f'{p.stem}_result.jpg', im0)
                        else:  # 'video' or 'stream'
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

            # Print time (inference-only)
        avertime += 1
        sumtime = sumtime + (dt[3].dt * 1E3)
        avertime_data = sumtime / avertime
        FPS = (1000 / avertime_data)
        LOGGER.info(f"{s}{'' if len(det) else '(no detections), '}{dt[3].dt * 1E3:.1f}ms, aver_time:{avertime_data}")
        if avertime >= 10:
            avertime = 0
            sumtime = 0
            with open(f"speed.txt", "a") as f:
                f.write(f"平均时间:{avertime_data:2f}" + "\n")
        if readValue != 0:
            print(f"读数处理时间:{dt[0].dt * 1E3:.1f}ms")
        print(f"过程预测时间(包含图像检测读数时间):{dt[2].dt * 1E3:.1f}ms")
        print(f"总时间:{dt[3].dt * 1E3:.1f}ms")

    # Print results
    t = tuple(x.t / seen * 1E3 for x in dt)  # speeds per image
    LOGGER.info(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {(1, 3, *imgsz)}' % t)
    if save_txt or save_img:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        LOGGER.info(f"Results saved to {colorstr('bold', save_dir)}{s}")
    if update:
        strip_optimizer(weights[0])  # update model (to fix SourceChangeWarning)


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--angle', type=float, default=267, help='tne range angle of the plate')
    parser.add_argument('--range' , type=float, default=100, help='range of the plate')
    parser.add_argument('--weights', nargs='+', type=str, default='runs/train/100-16-v5n-7/weights/100-16-v5n-7.pt', help='model path or triton URL')
    parser.add_argument('--source', type=str, default=0, help='file/dir/URL/glob/screen/0(webcam)')
    parser.add_argument('--data', type=str, default=ROOT / 'data/coco128.yaml', help='(optional) dataset.yaml path')
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640], help='inference size h,w')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='NMS IoU threshold')
    parser.add_argument('--max-det', type=int, default=1000, help='maximum detections per image')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='show results')
    parser.add_argument('--save-txt', action='store_false', help='save results to *.txt')
    parser.add_argument('--save-csv', action='store_true', help='save results in CSV format')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--save-crop', action='store_false', help='save cropped prediction boxes')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --classes 0, or --classes 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--visualize', action='store_true', help='visualize features')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default=ROOT / 'runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--line-thickness', default=3, type=int, help='bounding box thickness (pixels)')
    parser.add_argument('--hide-labels', default=False, action='store_true', help='hide labels')
    parser.add_argument('--hide-conf', default=False, action='store_true', help='hide confidences')
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    parser.add_argument('--dnn', action='store_true', help='use OpenCV DNN for ONNX inference')
    parser.add_argument('--vid-stride', type=int, default=1, help='video frame-rate stride')
    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand
    print_args(vars(opt))
    return opt


def main(opt):
    check_requirements(ROOT / 'requirements.txt', exclude=('tensorboard', 'thop'))
    run(**vars(opt))


if __name__ == '__main__':
    opt = parse_opt()
    main(opt)
