from models.yolo.model import YOLO
import argparse
from pathlib import Path
import os
import sys
FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv8 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

def detect(weights,source,conf,iou,imgsz,half,device,max_det,vid_stride,visualize,augment,agnostic_nms,classes,show,
        save, save_frames, save_txt, save_crop,show_label,show_conf,show_boxes,line_width):
    model = YOLO(weights)
    res = model.predict(source=source, conf=conf, iou=iou, imgsz=imgsz, half=half, device=device,max_det=max_det,
                        vid_stride=vid_stride, visualize=visualize, augment=augment, agnostic_nms=agnostic_nms,
                        classes=classes, show=show,save=save, save_frames=save_frames, save_txt=save_txt, save_crop=save_crop,
                        show_labels=show_label, show_conf=show_conf,show_boxes=show_boxes,line_width=line_width)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default='yolov8s.pt', help='weight path')
    parser.add_argument('--source', type=str, default=ROOT / 'assets')
    parser.add_argument('--conf', type=float, default=0.25)
    parser.add_argument('--iou', type=float, default=0.45)
    parser.add_argument('--imgsz', type=int, default=640)
    parser.add_argument('--half', action='store_true', help='half detect')
    parser.add_argument('--device', type=str, default='cuda', help='device cuda or cpu')
    parser.add_argument('--max_det', type=int, default=300)
    parser.add_argument('--vid_stride', type=int, default=1)
    parser.add_argument('--visualize', action='store_true', help='feat visualize')
    parser.add_argument('--augment', action='store_true')
    parser.add_argument('--agnostic_nms', action='store_true')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --classes 0, or --classes 0 2 3')
    # Visualization arguments
    parser.add_argument('--show', action='store_true', help='displays the annotated images or videos in a window')
    parser.add_argument('--save', action='store_true', help='save results')
    parser.add_argument('--save_frames', action='store_true', help='saves individual frames as images')
    parser.add_argument('--save_txt', action='store_true', help='save results as txt')
    parser.add_argument('--save_crop', action='store_true', help='save crop')
    parser.add_argument('--save_conf', action='store_true', help='save conf')
    parser.add_argument('--show_label', action='store_false', help='Displays labels for each detection in the visual output')
    parser.add_argument('--show_conf', action='store_false', help='Displays the confidence score for each detection alongside the label')
    parser.add_argument('--show_boxes', action='store_false', help='Draws bounding boxes around detected objects')
    parser.add_argument('--line_width', type=int, default=2, help='Specifies the line width of bounding boxes')
    args = parser.parse_args()
    print(args)
    detect(args.weights, args.source, args.conf, args.iou, args.imgsz, args.half, args.device, args.max_det,
           args.vid_stride, args.visualize, args.augment, args.agnostic_nms, args.classes, args.show, args.save,
           args.save_frames, args.save_txt, args.save_crop, args.show_label, args.show_conf, args.show_boxes,args.line_width)