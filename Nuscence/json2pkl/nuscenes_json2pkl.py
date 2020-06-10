import os
import sys

import numpy as np
from utils.utils import iou3d, convert_3dbox_to_8corner
from sklearn.utils.linear_assignment_ import linear_assignment

from nuscenes import NuScenes
from nuscenes.eval.common.config import config_factory
from nuscenes.eval.tracking.evaluate import TrackingEval
from nuscenes.eval.detection.data_classes import DetectionConfig
from nuscenes.eval.detection.data_classes import DetectionBox
from nuscenes.eval.tracking.data_classes import TrackingBox
from nuscenes.eval.common.loaders import load_prediction, load_gt, add_center_dist, filter_eval_boxes
from nuscenes.eval.tracking.loaders import create_tracks
from pyquaternion import Quaternion

import IPython
import argparse
import pickle

import json as JSON
import copy

# Settings.
parser = argparse.ArgumentParser(description='Get nuScenes stats.',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('-e', '--eval_set', type=str, default='val',
                    help='Which dataset split to evaluate on, train, val or test.')
# parser.add_argument('--verbose', type=int, default=1,
#                     help='Whether to print to stdout.')
parser.add_argument('-gt', '--label', action='store_true', help='Whether to save gt label or prediction.')
parser.add_argument('-CT', '--confidence_threshold', type=float, default=0.2)

args = parser.parse_args()

LABEL = args.label
PREDICT = not LABEL
eval_set_ = args.eval_set # 'Which dataset split to evaluate on, train, val or test.'

type_whitelist = ('Car', 'Pedestrian', 'car', 'pedestrian', 'bicycle')


class DetectionDataObject:
    def __init__(self, seq, frame):
        self.object = {}
        self.object['metadata'] = {}
        self.object['metadata']['seq_idx'] = seq
        self.object['metadata']['frame_idx'] = frame
        self.object['sample_token'] = []
        self.object['name'] = []
        self.object['track_id'] = np.array([])
        self.object['truncated'] = np.array([])
        self.object['occluded'] = np.array([])
        self.object['alpha'] = np.array([])
        self.object['bbox'] = np.zeros((0, 4))
        self.object['dimensions'] = np.zeros((0, 3))
        self.object['location'] = np.zeros((0, 3))
        self.object['global_location'] = np.zeros((0, 3))
        self.object['rotation_y'] = np.array([])
        self.object['score'] = np.array([])

    def get_object(self):
        return self.object


def rotation_to_positive_z_angle(rotation):
    q = Quaternion(rotation)
    angle = q.angle if q.axis[2] > 0 else -q.angle
    return angle


def sort_seq(data, seq_length):
    pointer = 0 # save the sorted data into new data, use this pointer to move on in the new data
    sorted_data = copy.deepcopy(data)
    for seq_num in range(seq_length):
        for i in range(len(data)):
            data_i = data[i]
            if data_i['metadata']['seq_idx'] == seq_num:
                sorted_data[pointer] = data_i
                pointer += 1

    # IPython.embed()
    # for i in range(pointer):
    #     s = sorted_data[i]['metadata']['seq_idx']
    #     f = sorted_data[i]['metadata']['frame_idx']
    #     print('seq: %d, frame %d: ' % (s, f))
    return sorted_data


if __name__ == '__main__':
    verbose_ = True

    json = '/data/Nuscences/Detection_results/detection-megvii/megvii_' + eval_set_ + '.json'
    # json = '/data/Nuscences/Detection_results/detection-megvii/megvii_test.json'
    # json = '/data/Nuscences/Detection_results/detection-megvii/megvii_val.json'

    # with open(json) as f:
    #     data = JSON.load(f)
    # IPython.embed()


    # get root data
    if eval_set_ != 'test':
        gt_root = '/data/Nuscences/v1.0-trainval_meta'
        version = 'v1.0-trainval'
        nusc = NuScenes(version=version, dataroot=gt_root, verbose=True)
    else:
        gt_root = '/data/Nuscences/v1.0-test_meta'
        version = 'v1.0-test'
        nusc = NuScenes(version=version, dataroot=gt_root, verbose=True)

    if PREDICT:
        # get predicted bboxes and add ego position into bboxes
        pred_boxes, _ = load_prediction(json, 10000, DetectionBox)
        pred_boxes = add_center_dist(nusc, pred_boxes)
        boxes = pred_boxes
        print('compose predicted data')

    if LABEL:
        # get gt bboxes and add ego position into bboxes

        gt_boxes = load_gt(nusc, eval_set_, TrackingBox)
        gt_boxes = add_center_dist(nusc, gt_boxes)
        boxes = gt_boxes
        print('compose label data.')
    # IPython.embed()
    # get seq name list and frame timestamp list
    seq_name_list = []
    seq_idx_list = []
    for idx, sample_token in enumerate(boxes.sample_tokens):
        # get scene name and seq_idx
        sample_record = nusc.get('sample', sample_token)
        scene_name = nusc.get('scene', sample_record['scene_token'])['name']
        if scene_name not in seq_name_list:
            seq_name_list.append(scene_name)
        seq_idx_list.append(seq_name_list.index(scene_name))

    frame_timestamp_list = []
    for s_i in range(len(seq_name_list)):
        frame_timestamp_list.append([])
    for idx, sample_token in enumerate(boxes.sample_tokens):
        sample_record = nusc.get('sample', sample_token)
        # get timestamp and frame_idx
        time_stamp = sample_record['timestamp']
        seq_idx = seq_idx_list[idx]
        frame_timestamp_list[seq_idx].append(time_stamp)

    # prepare the tracking data and save path
    tracking_data = []
    if PREDICT:
        save_path = '/data/Nuscences/Detection_results/detection-megvii-pkl/' + json.split('/')[-1].split('.')[0] + '.pkl'
    if LABEL:
        # if compose gt
        save_path = '/data/Nuscences/gt_tracking_pkl/label_' + args.eval_set + '.pkl'
        track_id_name_list = []
    # compose each frame
    for idx, sample_token in enumerate(boxes.sample_tokens):
        # each frame's bboxes
        if PREDICT:
            bboxes_frame = pred_boxes.boxes[sample_token]
        elif LABEL:
            bboxes_frame = gt_boxes.boxes[sample_token]

        # get seq_idx
        seq_idx = seq_idx_list[idx]
        # get frame_idx
        time_stamp = nusc.get('sample', sample_token)['timestamp']
        frame_idx = frame_timestamp_list[seq_idx].index(time_stamp)
        # print('seq: %d, frame %d: ' % (seq_idx, frame_idx))

        obj_c = DetectionDataObject(seq_idx, frame_idx)
        obj_i = obj_c.get_object()
        for b_i in bboxes_frame:
            if PREDICT:
                name = b_i.detection_name
                score = b_i.detection_score
            if LABEL:
                name = b_i.tracking_name
                score = 1.0
            if score < args.confidence_threshold:
                continue
            if name in type_whitelist:
                obj_i['sample_token'] = np.append(obj_i['sample_token'], b_i.sample_token)
                obj_i['name'] = np.append(obj_i['name'], name)
                # track ID
                if PREDICT:
                    obj_i['track_id'] = np.append(obj_i['track_id'], -1)
                if LABEL:
                    if b_i.tracking_id not in track_id_name_list:
                        track_id_name_list.append(b_i.tracking_id)
                    track_id = track_id_name_list.index(b_i.tracking_id)
                    obj_i['track_id'] = np.append(obj_i['track_id'], track_id)
                obj_i['truncated'] = np.append(obj_i['truncated'], -1)
                obj_i['occluded'] = np.append(obj_i['occluded'], -1)
                obj_i['alpha'] = np.append(obj_i['alpha'], -1)
                obj_i['bbox'] = np.append(obj_i['bbox'], [[-1, -1, -1, -1]], axis=0)
                obj_i['dimensions'] = np.append(obj_i['dimensions'], [[b_i.size[2], b_i.size[0], b_i.size[1]]], axis=0)
                obj_i['location'] = np.append(obj_i['location'], [[b_i.ego_translation[0], b_i.ego_translation[1], b_i.ego_translation[2]]], axis=0)
                obj_i['global_location'] = np.append(obj_i['global_location'], [[b_i.translation[0], b_i.translation[1], b_i.translation[2]]], axis=0)
                obj_i['rotation_y'] = np.append(obj_i['rotation_y'], rotation_to_positive_z_angle(b_i.rotation))
                obj_i['score'] = np.append(obj_i['score'], score)
        tracking_data.append(obj_i)

    tracking_data = sort_seq(tracking_data, len(seq_name_list))

    # save pkl
    print("save pkl into ", save_path)
    with open(save_path, 'wb') as f:
        pickle.dump(tracking_data, f)

    IPython.embed()



