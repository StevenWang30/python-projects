import os
import sys

import numpy as np
# from utils.utils import iou3d, convert_3dbox_to_8corner
from sklearn.utils.linear_assignment_ import linear_assignment

import json
import IPython
import argparse
import pickle

# Settings.
parser = argparse.ArgumentParser(description='Get nuScenes stats.',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('-e', '--eval_set', type=str, default='val',
                    help='Which dataset split to evaluate on, train, val or test.')
# parser.add_argument('--verbose', type=int, default=1,
#                     help='Whether to print to stdout.')
parser.add_argument('-gt', '--label', action='store_true', help='Whether to save gt label or prediction.')
parser.add_argument('-CT', '--confidence_threshold', type=float, default=0.1)

args = parser.parse_args()

LABEL = args.label
PREDICT = not LABEL
eval_set_ = args.eval_set # 'Which dataset split to evaluate on, train, val or test.'

type_whitelist = ['PEDESTRIAN', 'BICYCLE', 'MOTORBIKE', 'GOLF CAR', 'TRUCK', 'MOTORCYCLIST', 'CAR', 'FORKLIFT', 'CYCLIST']


class DetectionDataObject:
    def __init__(self, seq, frame):
        self.object = {}
        self.object['metadata'] = {}
        self.object['metadata']['seq_idx'] = seq
        self.object['metadata']['frame_idx'] = frame
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


if __name__ == '__main__':
    verbose_ = True

    json_root = '/data/yiqing_fushikang_tracking/label' #/0_bin.json'

    # prepare the tracking data and save path
    tracking_data = []
    save_path = '/data/yiqing_fushikang_tracking/tracking_pkl/tracking_label.pkl'

    seq_list = [[0,120],
                [121, 218],
                [219, 500],
                [501, 620],
                [621, 1262],
                [1263, 1379],
                [1380, 1871]]  #save each seq start and end.

    for s in range(len(seq_list)):
        seq_start = seq_list[s][0]
        seq_end = seq_list[s][1]
        seq_idx = s
        for i in range(seq_start, seq_end):
            json_file = os.path.join(json_root, str(i) + '_bin.json')
            with open(json_file, 'r') as j:
                detections = json.loads(j.read())

            frame_idx = i - seq_start
            obj_c = DetectionDataObject(seq_idx, frame_idx)
            obj_i = obj_c.get_object()
            for b in range(len(detections['elem'])):
                name = detections['elem'][b]['class']
                if name not in type_whitelist:
                    type_whitelist.append(name)
                obj_i['name'] = np.append(obj_i['name'], name)
                # track ID
                obj_i['track_id'] = np.append(obj_i['track_id'], detections['elem'][b]['id'])
                obj_i['truncated'] = np.append(obj_i['truncated'], -1)
                obj_i['occluded'] = np.append(obj_i['occluded'], -1)
                obj_i['alpha'] = np.append(obj_i['alpha'], -1)
                obj_i['bbox'] = np.append(obj_i['bbox'], [[-1, -1, -1, -1]], axis=0)
                dim = [detections['elem'][b]['size']['depth'], detections['elem'][b]['size']['width'], detections['elem'][b]['size']['height']]
                obj_i['dimensions'] = np.append(obj_i['dimensions'], [dim], axis=0)
                loc = [detections['elem'][b]['position']['x'], detections['elem'][b]['position']['y'], detections['elem'][b]['position']['z']]
                obj_i['location'] = np.append(obj_i['location'], [loc], axis=0)
                obj_i['rotation_y'] = np.append(obj_i['rotation_y'], detections['elem'][b]['yaw'])
                obj_i['score'] = np.append(obj_i['score'], 1.0)
            obj_i['global_location'] = obj_i['location']
            tracking_data.append(obj_i)

    # save pkl
    with open(save_path, 'wb') as f:
        pickle.dump(tracking_data, f)
        print('save pkl into ', save_path)

    print('type_whitelist: ', type_whitelist)
    IPython.embed()


