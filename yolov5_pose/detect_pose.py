
import cv2
import torch
import torch.backends.cudnn as cudnn
import numpy as np
from numpy import random
import time
from yolov5_pose.models.experimental import attempt_load
from yolov5_pose.utils.datasets import LoadStreams, LoadImages, letterbox
from yolov5_pose.utils.general import check_img_size, non_max_suppression, scale_coords
from yolov5_pose.utils.plots import plot_skeleton_kpts
from torch.autograd import Variable
import torch.nn as nn
import time
from pathlib import Path
import sys

# ----------- DON'T TOUCH HERE -------------------
FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]
# fix change name folder
sys.path.insert(0, str(ROOT))
# -----------------------------------------------


class Y5Detect:
    def __init__(self, weights):
        """
        params weights: 'yolov7.pt'
        """
        self.weights = weights
        self.model_image_size = 640
        self.conf_threshold = 0.2
        self.iou_threshold = 0.45
        with torch.no_grad():
            self.model, self.device = self.load_model(use_cuda=True)
            self.stride = int(self.model.stride.max())  # model stride
            # self.model = nn.DataParallel(self.model, device_ids=None)
            self.image_size = check_img_size(self.model_image_size, s=self.stride)
            self.half = True if self.device == "cuda:0" else False
            if self.half:
                self.model.half()
            self.class_names = self.model.module.names if hasattr(self.model, 'module') else self.model.names
            self.colors = [[random.randint(0, 255) for _ in range(3)] for _ in self.class_names]

    def load_model(self, use_cuda=False):
        if use_cuda:
            use_cuda = torch.cuda.is_available()
            cudnn.benchmark = True
        device = torch.device("cuda:0" if use_cuda else "cpu")
        model = attempt_load(self.weights, map_location=device)
        print('yolov5 running with {}'.format(device))
        return model, device

    def preprocess_image(self, image_rgb):
        img = letterbox(image_rgb.copy(), self.image_size, stride=self.stride, auto=False)[0]

        # Convert
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
        img = np.ascontiguousarray(img)
        img = torch.from_numpy(img).to(self.device)
        img = img.half() if self.half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)
        return img

    def predict(self, image_rgb):
        with torch.no_grad():
            image_rgb_shape = image_rgb.shape
            img = self.preprocess_image(image_rgb)
            pred = self.model(img)[0]
            # apply non_max_suppression
            pred = non_max_suppression(pred, self.conf_threshold, self.iou_threshold, kpt_label=True)
            bboxes = []
            labels = []
            scores = []
            lables_id = []
            kpts = []
            scores_pt = []
            line = []
            for i, det in enumerate(pred):
                if det is not None and len(det):
                    # Rescale boxes from img_size to im0 size
                    det[:, :4] = scale_coords(img.shape[2:], det[:, :4], image_rgb_shape, kpt_label=False).round()
                    det[:, 6:] = scale_coords(img.shape[2:], det[:, 6:], image_rgb_shape, kpt_label=True, step=3).round()

                for det_idx, (*xyxy, conf, cls) in enumerate(reversed(det[:, :6])):
                    x1 = xyxy[0].cpu().data.numpy()
                    y1 = xyxy[1].cpu().data.numpy()
                    x2 = xyxy[2].cpu().data.numpy()
                    y2 = xyxy[3].cpu().data.numpy()
                    #                        print('[INFO] bbox: ', x1, y1, x2, y2)
                    bboxes.append(list(map(int, [x1, y1, x2, y2])))
                    label = self.class_names[int(cls)]
                    #                        print('[INFO] label: ', label)
                    labels.append(label)
                    lables_id.append(cls.cpu())
                    score = conf.cpu().data.numpy()
                    #                        print('[INFO] score: ', score)
                    scores.append(float(score))
                    kpt = det[det_idx, 6:].cpu().data.numpy()
                    steps = 3
                    num_kpts = len(kpt) // steps
                    point, score_pt, line_pt = [], [], []
                    for kid in range(num_kpts):
                        x_coord, y_coord = kpt[steps * kid], kpt[steps * kid + 1]
                        if steps == 3:
                            conf = kpt[steps * kid + 2]
                        point.append([int(x_coord), int(y_coord)])
                        score_pt.append(conf)
                    kpts.append(point)
                    scores_pt.append(score_pt)
                    # ----------- get line -------------
                    skeleton = [[16, 14], [14, 12], [17, 15], [15, 13], [12, 13], [6, 12],
                                [7, 13], [6, 7], [6, 8], [7, 9], [8, 10], [9, 11], [2, 3],
                                [1, 2], [1, 3], [2, 4], [3, 5], [4, 6], [5, 7]]

                    for sk_id, sk in enumerate(skeleton):
                        pos1 = (int(kpt[(sk[0] - 1) * steps]), int(kpt[(sk[0] - 1) * steps + 1]))
                        pos2 = (int(kpt[(sk[1] - 1) * steps]), int(kpt[(sk[1] - 1) * steps + 1]))
                        line_pt.append([pos1, pos2])
                    line.append(line_pt)

            return bboxes, labels, scores, lables_id, kpts, scores_pt, line

    def run_video(self, source):
        with torch.no_grad():
            webcam = source.isnumeric() or source.endswith('.txt') or source.lower().startswith(
                ('rtsp://', 'rtmp://', 'http://', 'https://'))
            if webcam:
                dataset = LoadStreams(source, img_size=self.image_size, stride=self.stride)
            else:
                dataset = LoadImages(source, img_size=self.image_size, stride=self.stride)
            for _, img, im0, vid_cap in dataset:
                start = time.time()
                img = torch.from_numpy(img).to(self.device)
                img = img.half() if self.half else img.float()  # uint8 to fp16/32
                img /= 255.0  # 0 - 255 to 0.0 - 1.0
                if img.ndimension() == 3:
                    img = img.unsqueeze(0)
                pred = self.model(img)[0]

                # Apply NMS
                pred = non_max_suppression(pred, self.conf_threshold, self.iou_threshold, kpt_label=True)

                bboxes = []
                labels = []
                scores = []
                lables_id = []
                kpts = []
                scores_pt = []
                line = []
                for i, det in enumerate(pred):
                    if det is not None and len(det):
                        # Rescale boxes from img_size to im0 size
                        det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape, kpt_label=False).round()
                        det[:, 6:] = scale_coords(img.shape[2:], det[:, 6:], im0.shape, kpt_label=True,
                                                  step=3).round()
                    for det_idx, (*xyxy, conf, cls) in enumerate(reversed(det[:, :6])):
                        x1 = xyxy[0].cpu().data.numpy()
                        y1 = xyxy[1].cpu().data.numpy()
                        x2 = xyxy[2].cpu().data.numpy()
                        y2 = xyxy[3].cpu().data.numpy()
                        #                        print('[INFO] bbox: ', x1, y1, x2, y2)
                        bboxes.append(list(map(int, [x1, y1, x2, y2])))
                        label = self.class_names[int(cls)]
                        #                        print('[INFO] label: ', label)
                        labels.append(label)
                        lables_id.append(cls.cpu())
                        score = conf.cpu().data.numpy()
                        #                        print('[INFO] score: ', score)
                        scores.append(float(score))
                        kpt = det[det_idx, 6:].cpu().data.numpy()

                        steps = 3
                        num_kpts = len(kpt) // steps
                        point, score_pt, line_pt = [], [], []
                        for kid in range(num_kpts):
                            x_coord, y_coord = kpt[steps * kid], kpt[steps * kid + 1]
                            if steps == 3:
                                conf = kpt[steps * kid + 2]
                            point.append([int(x_coord), int(y_coord)])
                            score_pt.append(conf)
                        kpts.append(point)
                        scores_pt.append(score_pt)
                        # ----------- get line -------------
                        skeleton = [[16, 14], [14, 12], [17, 15], [15, 13], [12, 13], [6, 12],
                                    [7, 13], [6, 7], [6, 8], [7, 9], [8, 10], [9, 11], [2, 3],
                                    [1, 2], [1, 3], [2, 4], [3, 5], [4, 6], [5, 7]]

                        for sk_id, sk in enumerate(skeleton):
                            pos1 = (int(kpt[(sk[0] - 1) * steps]), int(kpt[(sk[0] - 1) * steps + 1]))
                            pos2 = (int(kpt[(sk[1] - 1) * steps]), int(kpt[(sk[1] - 1) * steps + 1]))
                            line_pt.append([pos1, pos2])
                        line.append(line_pt)
                for idx, box in enumerate(bboxes):
                    icolor = self.class_names.index(labels[idx])
                    draw_boxes(im0, box, labels[idx], round(scores[idx] * 100), self.colors[icolor])
                draw_kpts(im0, kpts, line)
                fps = int(1 / (time.time() - start))
                cv2.putText(im0, 'FPS:' + str(fps), (0, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2, cv2.LINE_AA)
                cv2.imshow('video', im0)
                cv2.waitKey(1)

        return bboxes, labels, scores, lables_id, kpts, scores_pt, line


def draw_boxes(image, boxes, label, scores=None, color=None):
    if color is None:
        color = (0, 255, 0)
    xmin, ymin, xmax, ymax = boxes
    cv2.rectangle(image, (xmin, ymin), (xmax, ymax), color, 2)
    cv2.putText(image, label + "-{:d}".format(scores), (xmin, ymin), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
    return image, boxes


def draw_kpts(image, kpts, lines):

    for kpt in kpts:
        for idx, point in enumerate(kpt):
            cv2.circle(image, (point[0], point[1]), 5, (0, 0, 255), -1)
    for line in lines:
        for ln in line:
            cv2.line(image, ln[0], ln[1], (255, 0, 0), thickness=2)
    return image


if __name__ == '__main__':
    path_models = '/home/duyngu/Desktop/edgeai-yolov5-yolo-pose/weights/last.pt'
    url = '/home/duyngu/Downloads/video_test/TownCentre.mp4'
    y5_model = Y5Detect(weights=path_models)
    y5_model.run_video(url)
