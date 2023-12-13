import argparse
import json
import os

import cv2
import numpy as np
import ultralytics
from addict import Dict


def gather_confmatrix(predicts, targets):
    matrix = Dict(tp=0, fp=0, fn=0, tn=0)
    for pred, target in zip(predicts, targets):
        if target == 1:
            if pred == 1:
                matrix.tp += 1
            else:
                matrix.fn += 1
        else:
            if pred == 1:
                matrix.fp += 1
            else:
                matrix.tn += 1
    return matrix


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--video_path', type=str, dest="video_path", default="videos")
    parser.add_argument('--polygon_path', type=str, dest="polygon_path", default="polygons.json")
    parser.add_argument('--intervals_path', type=str, dest="intervals_path", default="time_intervals.json")

    args = parser.parse_args()

    model = ultralytics.YOLO("yolov8m.pt")

    with open(args.polygon_path) as f:
        poly = json.load(f)
    with open(args.intervals_path) as f:
        time_intervals = json.load(f)
    video_names = list(time_intervals.keys())

    for video_name in video_names:
        cap = cv2.VideoCapture(os.path.join(args.video_path, video_name))
        ret, image = cap.read()

        # initialize mask of target area where we don't want to see vehicles
        current_poly = np.array(poly[video_name], dtype=np.int32).reshape((-1, 1, 2))
        polymask = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
        polymask = cv2.fillPoly(polymask, [current_poly], color=1)

        positive_frames = []
        num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        for frame_num in range(num_frames):
            ret, image = cap.read()
            if not ret:
                continue

            # inference with YOLO
            detections = model(image,
                               imgsz=800,
                               verbose=False,
                               classes=[2, 5, 7],  # needed classes are 2/car, 5/bus, 7/truck
                               )

            # draw boxes of detected vehicles
            box_mask = np.zeros_like(polymask, dtype=np.uint8)
            for img in detections:
                for box in img.boxes:
                    coords = box.xyxy.cpu().int().squeeze().numpy()
                    box_mask = cv2.rectangle(box_mask, (coords[0], coords[1]), (coords[2], coords[3]), 1, -1)

            # calculate if any detected box overlaps with our desired area
            overlapped = (polymask * box_mask).sum() > 1
            if overlapped:
                positive_frames.append(frame_num)

        # encode as one-hot to calculate metrics
        onehot_result = [1 if x in positive_frames else 0 for x in range(num_frames)]

        # encode gt intervals as well
        video_intervals = time_intervals[video_name]
        target_frames = []
        for (x, y) in video_intervals:
            target_frames.extend(list(range(x, y + 1)))
        onehot_target = [1 if x in target_frames else 0 for x in range(num_frames)]

        # build confusion matrix for metrics calculation
        conf_matrix = gather_confmatrix(onehot_result, onehot_target)
        accuracy = (conf_matrix.tp + conf_matrix.tn) / (sum(conf_matrix.values()))
        precision = conf_matrix.tp / (conf_matrix.tp + conf_matrix.fp + 1e-8)
        recall = conf_matrix.tp / (conf_matrix.tp + conf_matrix.fn + 1e-8)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)
        print(f"{video_name}: Accuracy: {accuracy:.2f}, Precision: {precision:.2f}, Recall: {recall:.2f}, F1: {f1:.2f}")