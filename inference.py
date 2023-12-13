import argparse
import json
import os

import cv2
import numpy as np
import ultralytics


def transform_ranges(frames):
    if len(frames) == 0:
        return []
    result = []
    left = positive_frames[0]
    last = positive_frames[0]
    for i in range(1, len(positive_frames)):
        if positive_frames[i] - last != 1:
            if left != last:
                result.append([left, last])
            left = positive_frames[i]
            last = positive_frames[i]
        else:
            last = positive_frames[i]
    if left != last:
        result.append([left, last])
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--video_path', type=str, dest="video_path", required=True)
    parser.add_argument('--polygon_path', type=str, dest="polygon_path", required=True)
    parser.add_argument('--output_path', type=str, dest="output_path", required=True)

    args = parser.parse_args()

    model = ultralytics.YOLO("yolov8m.pt")
    cap = cv2.VideoCapture(args.video_path)
    video_name = os.path.basename(args.video_path)
    ret, image = cap.read()
    with open(args.polygon_path) as f:
        poly = json.load(f)[video_name] # not sure about key, so it is being inherited from file name

    # initialize mask of target area where we don't want to see vehicles
    current_poly = np.array(poly, dtype=np.int32).reshape((-1, 1, 2))
    polymask = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
    polymask = cv2.fillPoly(polymask, [current_poly], color=1)

    positive_frames = []
    num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    for frame_num in range(num_frames):
        box_mask = np.zeros_like(polymask, dtype=np.uint8)
        ret, image = cap.read()
        if not ret:
            continue

        # inference with YOLO
        detections = model(image,
                           imgsz=800,
                           conf=0.25,
                           verbose=False,
                           classes=[2, 5, 7],  # needed classes are 2/car, 5/bus, 7/truck
                           )

        # draw boxes of detected vehicles
        for img in detections:
            for box in img.boxes:
                coords = box.xyxy.cpu().int().squeeze().numpy()
                box_mask = cv2.rectangle(box_mask, (coords[0], coords[1]), (coords[2], coords[3]), 1, -1)

        # calculate if any detected box overlaps with our desired area
        overlapped = (polymask * box_mask).sum() > 1
        if overlapped:
            positive_frames.append(frame_num)

    # transform to suitable format
    result = transform_ranges(positive_frames)
    result_dict = {video_name: result}
    with open(args.output_path, 'w') as f:
        json.dump(result_dict, f)
