import os
import numpy as np
import IPython
import sys
from KITTI_dataset_utils.utils import *

def load_track_data(data_dir='/data/KITTI_object_tracking/training', seq=None, idx1=None):
    npy_file = os.path.join(data_dir, 'label_processed', "%04d" % seq, "%06d.npy" % idx1)
    dict_load = np.load(npy_file, allow_pickle=True).item()
    pc_f1 = dict_load['pc1']
    pc_f2 = dict_load['pc2']
    # classification_label_f1 = dict_load['classification_label_f1']
    # classification_label_f2 = dict_load['classification_label_f2']
    tracking_label = dict_load['tracking_label']
    # bboxes_label_f1 = dict_load['bboxes_label_f1']
    # bboxes_label_f2 = dict_load['bboxes_label_f2']

    # # extract ground points in the foreground points(should in preprocessing, but for facility, change it here)
    # # IPython.embed()
    # planefile_f1 = os.path.join(data_dir, 'ground', "%04d" % seq, "%06d.txt" % idx1)
    # _, ground_idx = extract_ground_points_from_plane_file(pc_f1, planefile_f1, 0.2)
    # classification_label_f1[ground_idx, 1] = 1
    # classification_label_f1[ground_idx, 0] = 0
    # tracking_label[ground_idx] = 0
    # planefile_f2 = os.path.join(data_dir, 'ground', "%04d" % seq, "%06d.txt" % (idx1 + 1))
    # _, ground_idx = extract_ground_points_from_plane_file(pc_f2, planefile_f2, 0.1)
    # classification_label_f2[ground_idx, 1] = 1
    # classification_label_f2[ground_idx, 0] = 0

    #
    # # # if self.training:
    # if self.augment_displacement > 0:
    #     # print("Data augmentation should not be here.")
    #     # print("Data augmentation should not be here.")
    #     # print("Data augmentation should not be here.")
    #     # bb_num = bboxes_label_f2['bboxes_num']
    #     displacement_label = np.random.rand(3) * self.augment_displacement
    #     displacement_label[2] = 0
    #     pc1, pc2, bboxes_label_f1, bboxes_label_f2, tracking_label = \
    #         data_augmentation(pc1, pc2, classification_label_f1, classification_label_f2, bboxes_label_f1,
    #                           bboxes_label_f2, tracking_label, disp_f2=displacement_label)

    return pc_f1, pc_f2, tracking_label