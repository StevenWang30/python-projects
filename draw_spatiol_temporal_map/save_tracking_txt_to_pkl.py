import glob
import os
import IPython
import pickle
import numpy as np


class TrackingDataObject:
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
        self.object['rotation_y'] = np.array([])

    def get_object(self):
        return self.object



# tracking_seq_txt_root = '/data/KITTI_object_tracking/results_PointRCNNTrackNet/pred'
# tracking_seq_txt_root = '/data/KITTI_object_tracking/training/label_02'
tracking_seq_txt_root = '/data/KITTI_object_tracking/spatio-temporal-map/test_KITTI_upload'
txt_files = glob.glob(os.path.join(tracking_seq_txt_root, "*.txt"))
txt_files.sort()

tracking_data = []
current_start = 0
for txt in txt_files:
    print("compose ", txt)
    with open(txt, 'r') as f:
        str_list = f.readlines()
    str_list = [itm.rstrip() for itm in str_list if itm != '\n']
    if len(str_list) == 0:
        continue
    seq = int(txt.split('/')[-1].split('.')[0])
    max_idx = int(str_list[-1].split(' ')[0])
    for i in range(max_idx + 1):
        obj = TrackingDataObject(seq, i)
        tracking_data.append(obj.get_object())

    for s in str_list:
        idx_f = current_start + int(s.split(' ')[0])
        tracking_data[idx_f]['name'] = np.append(tracking_data[idx_f]['name'], s.split(' ')[2])
        tracking_data[idx_f]['track_id'] = np.append(tracking_data[idx_f]['track_id'], int(s.split(' ')[1]))
        tracking_data[idx_f]['truncated'] = np.append(tracking_data[idx_f]['truncated'], int(s.split(' ')[3]))
        tracking_data[idx_f]['occluded'] = np.append(tracking_data[idx_f]['occluded'], int(s.split(' ')[4]))
        tracking_data[idx_f]['alpha'] = np.append(tracking_data[idx_f]['alpha'], float(s.split(' ')[5]))
        tracking_data[idx_f]['bbox'] = np.append(tracking_data[idx_f]['bbox'], [np.array([float(s.split(' ')[6]), float(s.split(' ')[7]),
                                                      float(s.split(' ')[8]), float(s.split(' ')[9])])], axis=0)
        tracking_data[idx_f]['dimensions'] = np.append(tracking_data[idx_f]['dimensions'], [np.array([float(s.split(' ')[10]), float(s.split(' ')[11]),
                                                      float(s.split(' ')[12])])], axis=0)
        tracking_data[idx_f]['location'] = np.append(tracking_data[idx_f]['location'], [np.array([float(s.split(' ')[13]), float(s.split(' ')[14]),
                                                      float(s.split(' ')[15])])], axis=0)
        tracking_data[idx_f]['rotation_y'] = np.append(tracking_data[idx_f]['rotation_y'], float(s.split(' ')[16]))
    current_start = len(tracking_data)

# save_path = '/data/KITTI_object_tracking/results_PointRCNNTrackNet/tracking_pkl/training_result.pkl'
# save_path = '/home/skwang/PYProject/draw_spatiol-temporal_map/pkl_data/testing_label_result.pkl'
save_path = '/home/skwang/PYProject/KITTI_tracking_visualization/KITTI_test_trajectory.pkl'
print("save pkl into ", save_path)
with open(save_path, 'wb') as f:
    pickle.dump(tracking_data, f)

