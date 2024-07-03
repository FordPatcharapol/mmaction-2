import argparse
import cv2
import json
import pandas as pd
import numpy as np
import pytz
import imageio
import time
import sys
from datetime import datetime
from ultralytics import YOLO
from operator import itemgetter
from mmaction.apis import init_recognizer, inference_recognizer

from helper.utils import norm_kpts, plot_one_box, plot_skeleton_kpts, load_model_ext
from shapely.geometry import Polygon, Point
from collections import defaultdict, deque

# package parameter
this = sys.modules[__name__]

this.frame_counter = 0
this.recording = False
this.video_writer = None

config_file = 'E:/Ford/CP_Match/prototype/mmaction2/work_dirs/tsn_custom_2024_06_26/tsn_custom.py'
checkpoint_file = 'E:/Ford/CP_Match/prototype/mmaction2/work_dirs/tsn_custom_2024_06_26/epoch_100.pth'
# video_file = './data/lotus/val/shoplifter7.mp4'
label_file = './custom_label_map.txt'
display_size = (640, 480)
col_names = [
    '0_X', '0_Y', '1_X', '1_Y', '2_X', '2_Y', '3_X', '3_Y', '4_X', '4_Y', '5_X', '5_Y',
    '6_X', '6_Y', '7_X', '7_Y', '8_X', '8_Y', '9_X', '9_Y', '10_X', '10_Y', '11_X', '11_Y',
    '12_X', '12_Y', '13_X', '13_Y', '14_X', '14_Y', '15_X', '15_Y', '16_X', '16_Y'
]

cls_lst = ['normal', 'pickup', 'shoplifter']

pose_class_history = {}

ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", type=str,
                choices=[
                    'yolov8n-pose', 'yolov8s-pose', 'yolov8m-pose'
                ],
                default='./models/detect/yolov8m-pose.pt',
                help="choose type of yolov8 pose model")
ap.add_argument("-mp", "--pose", type=str,
                default='./models/classify/datasets_2024-05-14_Half/model.h5',
                help="path to saved keras model")
ap.add_argument("-ms", "--shoplifter", type=str,
                default='./models/classify/dl_2024_05_16/best.pt',
                help="path to saved classify model")
ap.add_argument("-a", "--area", type=str,
                default='./polygons.txt',
                help="path to polygons area")
ap.add_argument("-ct", "--conft", type=float, default=0.75,
                help="conf of result model")
ap.add_argument("-cc", "--confc", type=float, default=0.5,
                help="conf of classify model")
ap.add_argument("-s", "--source", type=str, required=True,
                help="path to video/cam/RTSP")
args = vars(ap.parse_args())


# init model
model = init_recognizer(config_file, checkpoint_file,
                        device='cuda:0')  # or device='cuda:0'

# inferance
# pred_result = inference_recognizer(model, video_file)
# pred_scores = pred_result.pred_score.tolist()
# score_tuples = tuple(zip(range(len(pred_scores)), pred_scores))
# score_sorted = sorted(score_tuples, key=itemgetter(1), reverse=True)
# top_label = score_sorted[:2]

# labels = open(label_file).readlines()
# labels = [x.strip() for x in labels]

# results = [(labels[k[0]], k[1]) for k in top_label]

# print('The top labels with corresponding scores are:')
# print(results)


def pose_3d_action_recognition(filename):
    pred_result = inference_recognizer(model, filename)
    pred_scores = pred_result.pred_scores.item.tolist()
    score_tuples = tuple(zip(range(len(pred_scores)), pred_scores))
    score_sorted = sorted(score_tuples, key=itemgetter(1), reverse=True)
    top_label = score_sorted[:2]

    labels = open(label_file).readlines()
    labels = [x.strip() for x in labels]

    results = [(labels[k[0]], k[1]) for k in top_label]

    print('\n================== result ==================')
    # print(f'The top labels of {filename}: {results}\n')
    print(f'The top labels of {filename}: {results[0][0]}\n')
    # print(results)


def read_polygon():
    poly_lst = []
    thickness_list = []

    files = open(args['area'], "r")
    for data in files:
        parts = data.strip().replace(' ', '').split('),')
        parts = [part.replace('(', '').replace(')', '') for part in parts]
        polygon_pos = [tuple(map(int, part.split(','))) for part in parts]
        if polygon_pos[0] != polygon_pos[-1]:
            polygon_pos.append(polygon_pos[0])
        poly_lst.append(polygon_pos)
        thickness_list.append(1)

    files.close()
    poly_objects = [Polygon(p) for p in poly_lst]

    return poly_objects, poly_lst, thickness_list


def get_inference(img, model, model_shoplifter, saved_model, class_names, poly_objects, lst_polys, thickness_list, pose_history):
    thickness_list_copy = thickness_list.copy()
    source_img = img.copy()
    source_img = cv2.cvtColor(source_img, cv2.COLOR_BGR2RGB)

    results = model.track(
        img, verbose=False, conf=args['conft'], tracker="bytetrack.yaml", persist=True)
    objects_in_polygons = set()  # Set to track objects within polygons

    for result in results:
        for box, pose in zip(result.boxes, result.keypoints.data):
            lm_list = []

            if box.id is None:
                continue

            xmin, ymin, xmax, ymax, idx = int(box.xyxy[0][0]), int(
                box.xyxy[0][1]), int(box.xyxy[0][2]), int(box.xyxy[0][3]), int(box.id[0])
            bottom_center = Point(xmin + (xmax - xmin) / 2, ymax)
            crop_object_img = img[ymin:ymax, xmin:xmax]

            in_polygon = False
            for index, poly_in_lst in enumerate(poly_objects):
                if poly_in_lst.contains(bottom_center):
                    thickness_list_copy[index] = 3
                    in_polygon = True

            if in_polygon is False:
                continue

            for pnt in pose:
                x, y = pnt[:2]
                lm_list.append([int(x), int(y)])

            if len(lm_list) == 17:
                pre_lm = norm_kpts(lm_list)
                if pre_lm is None:
                    continue

                data = pd.DataFrame([pre_lm], columns=col_names)
                predict = saved_model.predict(data, verbose=False)[0]

                # model shoplifter
                results_shoplifter = model_shoplifter(
                    crop_object_img, verbose=False)

                results_bbox = cls_lst[results_shoplifter[0].probs.top1]

                if float(results_shoplifter[0].probs.top1conf) < 0.5:
                    results_bbox = 'normal'

                pose_class = 'normal'

                if max(predict) > args['confc']:
                    pose_class = class_names[predict.argmax()]

                if (pose_class == 'shoplifter' and results_bbox == 'shoplifter') or (pose_class == 'shoplifter' and max(predict) > args['confc']):
                    pose_class = 'shoplifter'
                elif results_bbox == 'shoplifter' and float(results_shoplifter[0].probs.top1conf) > 0.6:
                    pose_class = 'shoplifter'
                elif pose_class == 'pickup' and results_bbox == 'shoplifter':
                    pose_class = 'pickup'
                elif pose_class == 'shoplifter' and results_bbox == 'pickup':
                    pose_class = 'pickup'
                elif pose_class == 'pickup' or results_bbox == 'pickup':
                    pose_class = 'pickup'
                else:
                    pose_class = 'normal'

                if idx not in pose_history:
                    pose_history[idx] = deque(maxlen=5)
                pose_history[idx].append(pose_class)

                most_common_pose_class = max(
                    set(pose_history[idx]), key=pose_history[idx].count)

                plot_one_box(
                    box.xyxy[0], img, most_common_pose_class, f'ID:{idx} {most_common_pose_class}')
                plot_skeleton_kpts(img, pose, radius=3,
                                   line_thick=1, confi=0.5)

                if most_common_pose_class in ['pickup', 'shoplifter']:
                    if not this.recording:
                        this.recording = True
                        this.frame_counter = 0
                        now_time = datetime.fromtimestamp(
                            time.time(), tz=pytz.timezone('Asia/Bangkok'))
                        formatted_date = str(
                            now_time.strftime('%Y-%m-%d_%H-%M-%S'))
                        this.filename = f'./videos/result/{formatted_date}.mp4'
                        this.video_writer = imageio.get_writer(
                            this.filename, fps=30)
                    else:
                        this.video_writer.append_data(source_img)
                        this.frame_counter += 1

    # Update history for objects not in any polygons to keep their state continuous
    for obj_id in pose_history.keys():
        if obj_id not in objects_in_polygons:
            # You can choose an appropriate default state
            pose_history[obj_id].append('normal')

    for index, poly in enumerate(lst_polys):
        cv2.polylines(img, [np.array(poly)], True,
                      (0, 255, 255), thickness_list_copy[index])

    if this.frame_counter >= (30 * 2):
        this.recording = False
        this.video_writer.close()
        print("before inferance")
        pose_3d_action_recognition(this.filename)


def main():
    video_path = args['source']
    saved_model, meta_str = load_model_ext(args['pose'])
    class_names = json.loads(meta_str)
    model = YOLO(f"{args['model']}")
    model_shoplifter = YOLO(f"{args['shoplifter']}")
    poly_objects, lst_polys, thickness_list = read_polygon()
    pose_history = defaultdict(lambda: deque(maxlen=5))

    if video_path.isnumeric():
        video_path = int(video_path)
    cap = cv2.VideoCapture(video_path)

    while True:
        success, img = cap.read()
        if not success:
            break

        img = cv2.resize(img, display_size)
        get_inference(img, model, model_shoplifter, saved_model, class_names,
                      poly_objects, lst_polys, thickness_list, pose_history)

        cv2.imshow('img', img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    if this.video_writer is not None:
        this.video_writer.close()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
