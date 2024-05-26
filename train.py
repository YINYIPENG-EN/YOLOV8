# from ultralytics import YOLO
import argparse
import os.path

from models.yolo import YOLO
from pathlib import Path
import sys
FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # yolov8 root dir
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative
if __name__ == '__main__':
    parse = argparse.ArgumentParser('Yolov8')
    parse.add_argument('--weights', type=str, default='yolov8s', help='yolov8 weight path')
    parse.add_argument('--model', type=str, default='yolov8s.yaml', help='model')
    parse.add_argument('--data', type=str, default=ROOT / 'cfg/datasets/mydata.yaml', help='data path')
    parse.add_argument('--epochs', type=int, default=100, help='train epochs')
    parse.add_argument('--device', type=str, default='cuda', help='cuda or cpu')
    parse.add_argument('--cache', action='store_false', help='dataset cache')
    parse.add_argument('--workers', type=int, default=4, help='number of workers')
    parse.add_argument('--bs', type=int, default=64, help='batch size')
    parse.add_argument('--patience', type=int, default=50, help='EarlyStopping patience (epochs without improvement)')
    parse.add_argument('--imgsz', type=int, default=640, help='input shape')
    parse.add_argument('--save', action='store_false', help='save model')
    parse.add_argument('--save_period', type=int, default=-1, help='Save checkpoint every x epochs (disabled if < 1)')
    parse.add_argument('--project', default=ROOT / 'runs/train', help='save to project/name')
    parse.add_argument('--name', default='exp', help='save to project/name')
    parse.add_argument('--exist_ok', action='store_true', help='existing project/name ok, do not increment')
    parse.add_argument('--pretrained', action='store_true', default=False, help='pretrained weights')
    parse.add_argument('--optimizer', type=str, default='auto', choices=['SGD', 'Adam', 'AdamW'])
    parse.add_argument('--verbose', action='store_false', help='verbose')
    parse.add_argument('--seed', type=int, default=0)
    parse.add_argument('--single_cls', action='store_true')
    parse.add_argument('--rect', action='store_true')
    parse.add_argument('--cos_lr', action='store_true')
    parse.add_argument('--resume', action='store_true', help='resume train')
    parse.add_argument('--amp', action='store_false')
    parse.add_argument('--freeze', nargs='+', type=int, default=None, help='Freeze layers: backbone=10')
    args = parse.parse_args()
    print(args)

    model = YOLO(args.weights)
    res = model.train(model=args.model, data=args.data, epochs=args.epochs, batch=args.bs,
                      device=args.device, cache=args.cache, workers=args.workers, patience=args.patience, imgsz=args.imgsz,
                      save=args.save, save_period=args.save_period, project=args.project, name=args.name,
                      exist_ok=args.exist_ok, pretrained=args.pretrained,optimizer=args.optimizer,verbose=args.verbose,
                      seed=args.seed, single_cls=args.single_cls, rect=args.rect, cos_lr=args.cos_lr,resume=args.resume,
                      amp=args.amp,freeze=args.freeze)
