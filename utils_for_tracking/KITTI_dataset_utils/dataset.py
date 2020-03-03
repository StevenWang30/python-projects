import os
import numpy as np
# from sklearn.model_selection import train_test_split
import IPython
import sys
# BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# sys.path.append(BASE_DIR)
# sys.path.append(os.path.join(BASE_DIR, 'utils'))
from utils_for_tracking.KITTI_dataset_utils.utils import *
import open3d as o3d

# type2class={'Car':0, 'Van':1, 'Truck':2, 'Pedestrian':3,
#               'Person_sitting':4, 'Cyclist':5, 'Tram':6, 'Misc':7}
type2class={'Car':0, 'Van':1}
class2type = {type2class[t]:t for t in type2class}
type2onehotclass = {'Car': 0, 'Pedestrian': 1, 'Cyclist': 2}
type_whitelist = ('Car', 'Van')
# type2onehotclass = {'Car': 0, 'Pedestrian': 1, 'Cyclist': 2}

type_mean_size = {'Car': np.array([3.88311640418,1.62856739989,1.52563191462]),
                    'Van': np.array([5.06763659,1.9007158,2.20532825]),
                    'Truck': np.array([10.13586957,2.58549199,3.2520595]),
                    'Pedestrian': np.array([0.84422524,0.66068622,1.76255119]),
                    'Person_sitting': np.array([0.80057803,0.5983815,1.27450867]),
                    'Cyclist': np.array([1.76282397,0.59706367,1.73698127]),
                    'Tram': np.array([16.17150617,2.53246914,3.53079012]),
                    'Misc': np.array([3.64300781,1.54298177,1.92320313])}
NUM_HEADING_BIN = 12
NUM_SIZE_CLUSTER = len(type2class) # one cluster for each type
class_mean_size_arr = np.zeros((NUM_SIZE_CLUSTER, 3)) # size clustrs
for i in range(NUM_SIZE_CLUSTER):
    class_mean_size_arr[i,:] = type_mean_size[class2type[i]]


class LoadDataset():
    def __init__(self, root='/data/KITTI_object_tracking/training', aug_displacement=0.0, npoints=15000, split=0.6, train=True, shuffle=True, test=False, sequence=-1):
        if train == False:
            if sequence == -1:
                print("The test dataset must specify the sequence num.")
                assert True
            shuffle = False
        self.npoints = npoints
        self.training = train
        self.root = root
        self.type_whitelist = type_whitelist
        self.test_flag = test
        self.augment_displacement = aug_displacement
        vel_path = os.path.join(root, "velodyne")
        datapath = []
        test_datapath = []
        files = os.listdir(vel_path)
        files.sort(key=lambda x: int(x.split('.')[0]))
        for dir in files:
            dir_path = os.path.join(vel_path, dir)
            bin_files = os.listdir(dir_path)
            bin_files.sort(key=lambda x: int(x.split('.')[0]))
            for i in range(len(bin_files) - 1):
                datapath.append([os.path.join(vel_path, dir, bin_files[i]), os.path.join(vel_path, dir, bin_files[i + 1])])
                if test == True and int(dir) == sequence:
                    test_datapath.append([os.path.join(vel_path, dir, bin_files[i]), os.path.join(vel_path, dir, bin_files[i + 1])])
        [train_datapath, eval_datapath] = train_test_split(datapath, test_size=1 - split, shuffle=False)
        if self.training:
            if shuffle:
                np.random.shuffle(train_datapath)
            self.datapath = train_datapath
        else:
            self.datapath = eval_datapath
        if test:
            self.datapath = test_datapath

    def __getitem__(self, index):
        fn = self.datapath[index]
        labelfile = fn[0].replace('velodyne', 'label_processed').replace('bin', 'npy')
        dict_load = np.load(labelfile, allow_pickle=True).item()
        pc1 = dict_load['pc1']
        pc2 = dict_load['pc2']
        classification_label_f1 = dict_load['classification_label_f1']
        classification_label_f2 = dict_load['classification_label_f2']
        tracking_label = dict_load['tracking_label']
        bboxes_label_f1 = dict_load['bboxes_label_f1']
        bboxes_label_f2 = dict_load['bboxes_label_f2']

        # extract ground points in the foreground points(should in preprocessing, but for facility, change it here)
        # IPython.embed()
        planefile_f1 = fn[0].replace(".bin", ".txt").replace('velodyne', 'ground')
        _, ground_idx = extract_ground_points_from_plane_file(pc1, planefile_f1, 0.2)
        classification_label_f1[ground_idx, 1] = 1
        classification_label_f1[ground_idx, 0] = 0
        tracking_label[ground_idx] = 0
        planefile_f2 = fn[1].replace(".bin", ".txt").replace('velodyne', 'ground')
        _, ground_idx = extract_ground_points_from_plane_file(pc2, planefile_f2, 0.1)
        classification_label_f2[ground_idx, 1] = 1
        classification_label_f2[ground_idx, 0] = 0

        NUM_POINT_f1 = np.min((self.npoints, pc1.shape[0]))
        NUM_POINT_f2 = np.min((self.npoints, pc2.shape[0]))

        # shuffle and sample
        if self.training:
            # f1
            positive_idx = np.where(classification_label_f1[:, 0] == 1)
            negative_idx = np.where(classification_label_f1[:, 1] == 1)
            foreground_points = pc1[positive_idx]
            background_points = pc1[negative_idx]
            foreground_tracking = tracking_label[positive_idx]
            background_tracking = tracking_label[negative_idx]
            if foreground_points.shape[0] > NUM_POINT_f1 / 2:
                foreground_sample_idx = np.random.choice(foreground_points.shape[0], NUM_POINT_f1 // 2, replace=False)
                foreground_points = foreground_points[foreground_sample_idx]
                foreground_tracking = foreground_tracking[foreground_sample_idx]
            # if background_points.shape[0] == 0: IPython.embed()
            background_sample_idx = np.random.choice(background_points.shape[0], NUM_POINT_f1 - foreground_points.shape[0], replace=False)
            background_points = background_points[background_sample_idx]
            background_tracking = background_tracking[background_sample_idx]
            classification_label_f1 = np.append(np.array([[1, 0]]) + np.zeros((foreground_points.shape[0], 2)),
                                             np.array([[0, 1]]) + np.zeros((background_points.shape[0], 2)), axis=0)
            pc1 = np.append(foreground_points, background_points, axis=0)
            tracking_label = np.append(foreground_tracking, background_tracking, axis=0)
            idxs = np.arange(0, pc1.shape[0])
            np.random.shuffle(idxs)
            pc1 = pc1[idxs]
            classification_label_f1 = classification_label_f1[idxs]
            tracking_label = tracking_label[idxs]

            # f2
            positive_idx = np.where(classification_label_f2[:, 0] == 1)
            negative_idx = np.where(classification_label_f2[:, 1] == 1)
            foreground_points = pc2[positive_idx]
            background_points = pc2[negative_idx]
            if foreground_points.shape[0] > NUM_POINT_f2 / 2:
                foreground_sample_idx = np.random.choice(foreground_points.shape[0], NUM_POINT_f2 // 2, replace=False)
                foreground_points = foreground_points[foreground_sample_idx]
            background_sample_idx = np.random.choice(background_points.shape[0], NUM_POINT_f2 - foreground_points.shape[0],
                                                     replace=False)
            background_points = background_points[background_sample_idx]
            classification_label_f2 = np.append(np.array([[1, 0]]) + np.zeros((foreground_points.shape[0], 2)),
                                             np.array([[0, 1]]) + np.zeros((background_points.shape[0], 2)), axis=0)
            pc2 = np.append(foreground_points, background_points, axis=0)
            idxs = np.arange(0, pc2.shape[0])
            np.random.shuffle(idxs)
            pc2 = pc2[idxs]
            classification_label_f2 = classification_label_f2[idxs]
        else:
            # f1
            sample_idx = np.random.choice(pc1.shape[0], NUM_POINT_f1, replace=False)
            classification_label_f1 = classification_label_f1[sample_idx]
            pc1 = pc1[sample_idx]
            tracking_label = tracking_label[sample_idx]

            # f2
            sample_idx = np.random.choice(pc2.shape[0], NUM_POINT_f2, replace=False)
            classification_label_f2 = classification_label_f2[sample_idx]
            pc2 = pc2[sample_idx]

        # move to center
        pc_center = np.mean(pc1, 0)
        pc1 -= pc_center
        pc2 -= pc_center
        if bboxes_label_f1['bboxes_num'] != 0:
            bboxes_label_f1['bboxes_xyz'] -= pc_center
        if bboxes_label_f2['bboxes_num'] != 0:
            bboxes_label_f2['bboxes_xyz'] -= pc_center
        bboxes_label_f1['pc_center'] = pc_center
        bboxes_label_f2['pc_center'] = pc_center

        if self.npoints > NUM_POINT_f1:
            pc1 = np.append(pc1, np.array([[0, 0, -5]]) + np.zeros((self.npoints - NUM_POINT_f1, 3)), axis=0)
            classification_label_f1 = np.append(classification_label_f1, np.array([[0, 1]]) + np.zeros((self.npoints - NUM_POINT_f1, 2)), axis=0)
            tracking_label = np.append(tracking_label, np.zeros((self.npoints - NUM_POINT_f1, 7)), axis=0)
        if self.npoints > NUM_POINT_f2:
            pc2 = np.append(pc2, np.array([[0, 0, -5]]) + np.zeros((self.npoints - NUM_POINT_f2, 3)), axis=0)
            classification_label_f2 = np.append(classification_label_f2, np.array([[0, 1]]) + np.zeros((self.npoints - NUM_POINT_f2, 2)), axis=0)

        # # if self.training:
        if self.augment_displacement > 0:
            # print("Data augmentation should not be here.")
            # print("Data augmentation should not be here.")
            # print("Data augmentation should not be here.")
            # bb_num = bboxes_label_f2['bboxes_num']
            displacement_label = np.random.rand(3) * self.augment_displacement
            displacement_label[2] = 0
            pc1, pc2, bboxes_label_f1, bboxes_label_f2, tracking_label = \
                data_augmentation(pc1, pc2, classification_label_f1, classification_label_f2, bboxes_label_f1,
                                  bboxes_label_f2, tracking_label, disp_f2=displacement_label)

        NUM_POINT = self.npoints
        point_cloud = np.zeros((NUM_POINT * 2, 3))
        point_cloud[0:NUM_POINT] = pc1
        point_cloud[NUM_POINT:] = pc2
        return point_cloud, classification_label_f1, classification_label_f2, tracking_label, bboxes_label_f1, bboxes_label_f2

    def __len__(self):
        return len(self.datapath)


class LoadTestDataset():
    def __init__(self, root='/data/KITTI_object_tracking/testing', npoints=15000, sequence=-1):
        if sequence == -1:
            print("The test dataset must specify the sequence num.")
            assert True
        self.npoints = npoints
        self.root = root
        vel_path = os.path.join(root, "velodyne")
        datapath = []
        files = os.listdir(vel_path)
        files.sort(key=lambda x: int(x.split('.')[0]))
        for dir in files:
            if int(dir) == sequence:
                dir_path = os.path.join(vel_path, dir)
                bin_files = os.listdir(dir_path)
                bin_files.sort(key=lambda x: int(x.split('.')[0]))
                for i in range(len(bin_files) - 1):
                    datapath.append([os.path.join(vel_path, dir, bin_files[i]), os.path.join(vel_path, dir, bin_files[i + 1])])
        self.datapath = datapath

    def __getitem__(self, index):
        fn = self.datapath[index]
        sequence = fn[0].split('/')[-2]
        f1_n = fn[0]
        f2_n = fn[1]
        frame_1 = int(f1_n.split('/')[-1].split('.')[0])
        frame_2 = frame_1 + 1
        calibfile = os.path.join(self.root, "calib", sequence + ".txt")
        data, T_cam_velo = get_calibration(calibfile)

        # load point cloud
        pc1 = load_bin(f1_n)
        pc2 = load_bin(f2_n)

        # filter the data into the camera FO, since the dataset only label the points in the camera view.
        pc1 = filter_camera_angle(pc1)
        pc2 = filter_camera_angle(pc2)

        NUM_POINT_f1 = np.min((self.npoints, pc1.shape[0]))
        NUM_POINT_f2 = np.min((self.npoints, pc2.shape[0]))

        # f1
        sample_idx = np.random.choice(pc1.shape[0], NUM_POINT_f1, replace=False)
        pc1 = pc1[sample_idx]

        # f2
        sample_idx = np.random.choice(pc2.shape[0], NUM_POINT_f2, replace=False)
        pc2 = pc2[sample_idx]

        # move to center
        pc_center = np.mean(pc1, 0)
        pc1 -= pc_center
        pc2 -= pc_center

        if self.npoints > NUM_POINT_f1:
            pc1 = np.append(pc1, np.array([[0, 0, -5]]) + np.zeros((self.npoints - NUM_POINT_f1, 3)), axis=0)
        if self.npoints > NUM_POINT_f2:
            pc2 = np.append(pc2, np.array([[0, 0, -5]]) + np.zeros((self.npoints - NUM_POINT_f2, 3)), axis=0)

        NUM_POINT = self.npoints
        point_cloud = np.zeros((NUM_POINT * 2, 3))
        point_cloud[0:NUM_POINT] = pc1
        point_cloud[NUM_POINT:] = pc2
        return point_cloud, pc_center, T_cam_velo, fn

    def __len__(self):
        return len(self.datapath)



def batch_data_augmentation(pc1, pc2, batch_bboxes_label_f1, batch_bboxes_label_f2, batch_idx, tracking_label, T_cam_velo, disp_f1=None, disp_f2=None):
    if disp_f1 is not None:
        bb_num = batch_bboxes_label_f1['bboxes_num']
        if bb_num > 0:
            bboxes_xyz = batch_bboxes_label_f1['bboxes_xyz'][batch_idx]  # BB * 3
            bboxes_lwh = batch_bboxes_label_f1['bboxes_size'][batch_idx]  # BB * 3
            bboxes_angle = batch_bboxes_label_f1['heading_angles'][batch_idx]
            for bb in range(bb_num):
                if not np.any(bboxes_xyz[bb]):
                    break
                idx = find_points_idx_in_bbox(pc1, bboxes_xyz[bb], bboxes_lwh[bb], bboxes_angle[bb], T_cam_velo)
                if np.any(idx):
                    pc1[idx] += disp_f1
            # move bboxes
            batch_bboxes_label_f1['bboxes_xyz'][batch_idx] = batch_bboxes_label_f1['bboxes_xyz'][batch_idx] + disp_f1
            tracking_label[..., :3] -= disp_f1
    if disp_f2 is not None:
        bb_num = batch_bboxes_label_f2['bboxes_num']
        if bb_num > 0:
            bboxes_xyz = batch_bboxes_label_f2['bboxes_xyz'][batch_idx]  # BB * 3
            bboxes_lwh = batch_bboxes_label_f2['bboxes_size'][batch_idx]  # BB * 3
            bboxes_angle = batch_bboxes_label_f2['heading_angles'][batch_idx]
            for bb in range(bb_num):
                if not np.any(bboxes_xyz[bb]):
                    break
                idx = find_points_idx_in_bbox(pc2, bboxes_xyz[bb], bboxes_lwh[bb], bboxes_angle[bb], T_cam_velo)
                if np.any(idx):
                    pc2[idx] += disp_f2
            # move bboxes
            batch_bboxes_label_f2['bboxes_xyz'][batch_idx] = batch_bboxes_label_f2['bboxes_xyz'][batch_idx] + disp_f2
            tracking_label[..., :3] += disp_f2
    return pc1, pc2, batch_bboxes_label_f1, batch_bboxes_label_f2, tracking_label




def data_augmentation(pc1, pc2, classification_label_f1, classification_label_f2, bboxes_label_f1, bboxes_label_f2, tracking_label, disp_f1=None, disp_f2=None):
    if disp_f1 is not None:
        if bboxes_label_f1['bboxes_num'] > 0:
            # # do not augment the points in the bounding box
            # # just displace the whole frame's points in 1 meter
            # displacement = np.random.rand(3) * disp
            # displacement[2] = 0  # z axis displacement = 0
            # pc2 += displacement
            # bboxes_label_f1['bboxes_xyz'] += displacement
            # # pc1_positive = np.where(classification_label_f1[:,0] == 1)
            # tracking_label += displacement

            # augment the points in the bounding boxes
            bb_num = bboxes_label_f1['bboxes_num']
            # displacement_label = np.random.rand(bb_num, 3) * disp
            # displacement_label[:, 2] = 0
            # displacement_label = disp_f1

            bboxes_xyz = bboxes_label_f1['bboxes_xyz']  # BB * 3
            bboxes_lwh = bboxes_label_f1['bboxes_lwh']  # BB * 3
            bboxes_angle = bboxes_label_f1['heading_angle']

            for bb in range(bb_num):
                idx = find_points_idx_in_bbox(pc1, bboxes_xyz[bb], bboxes_lwh[bb], bboxes_angle[bb],
                                              bboxes_label_f1['T_cam_velo'])
                if np.any(idx):
                    pc1[idx] += disp_f1

            # move bboxes
            bboxes_label_f1['bboxes_xyz'] = bboxes_label_f1['bboxes_xyz'] + disp_f1
            tracking_label[..., :3] -= disp_f1
    if disp_f2 is not None:
        if bboxes_label_f2['bboxes_num'] > 0:
            # # do not augment the points in the bounding box
            # # just displace the whole frame's points in 1 meter
            # displacement = np.random.rand(3) * disp
            # displacement[2] = 0  # z axis displacement = 0
            # pc2 += displacement
            # bboxes_label_f2['bboxes_xyz'] += displacement
            # # pc1_positive = np.where(classification_label_f1[:,0] == 1)
            # tracking_label += displacement


            # augment the points in the bounding boxes
            bb_num = bboxes_label_f2['bboxes_num']
            # displacement_label = np.random.rand(bb_num, 3) * disp
            # displacement_label[:, 2] = 0
            # displacement_label = disp_f2

            bboxes_xyz = bboxes_label_f2['bboxes_xyz']  # BB * 3
            bboxes_lwh = bboxes_label_f2['bboxes_lwh']  # BB * 3
            bboxes_angle = bboxes_label_f2['heading_angle']

            for bb in range(bb_num):
                idx = find_points_idx_in_bbox(pc2, bboxes_xyz[bb], bboxes_lwh[bb], bboxes_angle[bb], bboxes_label_f2['T_cam_velo'])
                if np.any(idx):
                    pc2[idx] += disp_f2

            # move bboxes
            bboxes_label_f2['bboxes_xyz'] = bboxes_label_f2['bboxes_xyz'] + disp_f2
            tracking_label[..., :3] += disp_f2
            # print('add tracking and add bbox in the dataset.')
    # if disp_f1 is not None and disp_f2 is not None:
    #     tracking_label += disp_f1 - disp_f2
    return pc1, pc2, bboxes_label_f1, bboxes_label_f2, tracking_label


def preprocessing_object_tracking_label(pc1, pc2, label1, label2, T_cam_velo):
    tracking_label1 = process_tracking_label(label1, label2)
    tracking_classification_label_f1 = get_tracking_classification_label(pc1, tracking_label1, label2, T_cam_velo)
    classification_label_f2 = get_classification_label(pc2, label2, T_cam_velo)
    return tracking_classification_label_f1, classification_label_f2


def process_tracking_label(label1, label2):
    label1['next_track_id'] = -1 * np.ones(label1['bboxes_num'])
    for i in range(label1['bboxes_num']):
        track_id = label1['bboxes_trackID'][i]
        for j in range(label2['bboxes_num']):
            if track_id == label2['bboxes_trackID'][j]:
                label1['next_track_id'][i] = j
                break
    label1['next_track_id'] = label1['next_track_id'].astype(int)
    return label1


def get_tracking_classification_label(pointcloud, label1, label2, T_cam_velo):
    '''
    transform the point cloud to 0-1 label,
    0 means this point is foreground and
    1 means the point is the background.
    '''
    tracking_classification_label = np.zeros((pointcloud.shape[0], 5))
    if label1['bboxes_num'] == 0:
        return tracking_classification_label
    bboxes_xyz = label1['bboxes_xyz'] # BB * 3
    bboxes_lwh = label1['bboxes_lwh'] # BB * 3
    # bboxes_xyz_cam = [] # BB * 3
    # for i in range(label1['bboxes_num']):
    #     label_i = label1[i]
    #     xyz = np.array(label_i['t'])
    #     xyz_center_cam = xyz - [0, label_i['h'] / 2, 0]
    #     xyz_center = np.dot(T_cam_velo, np.append(xyz_center_cam, 1))[:3]
    #     lwh = np.array([label_i['l'], label_i['w'], label_i['h']])
    #     bboxes_xyz.append(xyz_center)
    #     bboxes_xyz_cam.append(xyz_center_cam)
    #     bboxes_lwh.append(lwh)

    dist2center = abs(np.expand_dims(pointcloud, 1) - np.expand_dims(bboxes_xyz, 0))  # N * BB * 3
    dist2center_norm = np.linalg.norm(dist2center, axis=-1)  # N * BB  # distance: x^2 + y^2 + z^2
    dist2center_norm_min = dist2center_norm.min(-1)  # N float norm
    dist2center_norm_min_index = np.argmin(dist2center_norm, axis=-1) # N index

    THRESHOLD = 0.3
    for i in range(pointcloud.shape[0]):
        bb_index = dist2center_norm_min_index[i]
        [l, w, h] = bboxes_lwh[bb_index]
        max = 1/2 * math.sqrt(l*l+w*w+h*h)
        if dist2center_norm_min[i] > max:
            tracking_classification_label[i][1] = 1
        else:
            xyz_dif = pointcloud[i] - bboxes_xyz[bb_index]
            xyz_rot = np.dot(roty4(-1 * label1['heading_angle'][bb_index]), np.dot(np.linalg.inv(T_cam_velo), np.append(xyz_dif, 1)))[:3]
            dist2center_rot = abs(xyz_rot)
            if dist2center_rot[0] <= l / 2+THRESHOLD and \
                    dist2center_rot[1] <= h / 2+THRESHOLD and \
                    dist2center_rot[2] <= w / 2+THRESHOLD:
                tracking_classification_label[i][0] = 1
                # judge if have next track object
                next_object = label1['next_track_id'][bb_index]
                if next_object != -1:
                    next_xyz = label2['bboxes_xyz'][next_object]
                    dxyz = next_xyz - bboxes_xyz[bb_index]
                    tracking_classification_label[i][-3:] = dxyz
            else:
                tracking_classification_label[i][1] = 1

    return tracking_classification_label


def get_classification_label(pointcloud, label, T_cam_velo):
    '''
    transform the point cloud to 0-1 label,
    0 means this point is foreground and
    1 means the point is the background.
    '''
    classification_label = np.zeros((pointcloud.shape[0], 2))
    if label['bboxes_num'] == 0:
        classification_label[:, 1] = 1
        return classification_label
    bboxes_xyz = label['bboxes_xyz'] # BB * 3
    bboxes_lwh = label['bboxes_lwh'] # BB * 3
    # for i in range(label['bboxes_num']):
    #     label_i = label[i]
    #     xyz = np.array(label_i['t'])
    #     xyz_center_cam = xyz - [0, label_i['h'] / 2, 0]
    #     xyz_center = np.dot(T_cam_velo, np.append(xyz_center_cam, 1))[:3]
    #     lwh = np.array([label_i['l'], label_i['w'], label_i['h']])
    #     bboxes_xyz.append(xyz_center)
    #     bboxes_lwh.append(lwh)

    dist2center = abs(np.expand_dims(pointcloud, 1) - np.expand_dims(bboxes_xyz, 0))  # N * BB * 3
    dist2center_norm = np.linalg.norm(dist2center, axis=-1)  # N * BB  # distance: x^2 + y^2 + z^2
    dist2center_norm_min = dist2center_norm.min(-1)  # N float norm
    dist2center_norm_min_index = np.argmin(dist2center_norm, axis=-1) # N index

    THRESHOLD = 0.3
    for i in range(pointcloud.shape[0]):
        bb_index = dist2center_norm_min_index[i]
        [l, w, h] = bboxes_lwh[bb_index]
        max = 1/2 * math.sqrt(l*l+w*w+h*h)
        if dist2center_norm_min[i] > max:
            classification_label[i][1] = 1
        else:
            xyz_dif = pointcloud[i] - bboxes_xyz[bb_index]
            xyz_rot = np.dot(roty4(-1 * label['heading_angle'][bb_index]), np.dot(np.linalg.inv(T_cam_velo), np.append(xyz_dif, 1)))[:3]
            dist2center_rot = abs(xyz_rot)
            if dist2center_rot[0] <= l / 2+THRESHOLD and \
                    dist2center_rot[1] <= h / 2+THRESHOLD and \
                    dist2center_rot[2] <= w / 2+THRESHOLD:
                classification_label[i][0] = 1
            else:
                classification_label[i][1] = 1

    return classification_label


def preprocessing_bboxes_label(self, label, T_cam_velo, datapath):
    bboxes_xyz = []
    bboxes_lwh = []
    semantic_labels = []
    heading_labels = []
    heading_residuals = []
    size_labels = []
    size_residuals = []
    heading_angles = []
    bboxes_trackID = []
    for i in range(len(label)):
        label_i = label[i]
        xyz = np.array(label_i['t'])
        xyz_center_cam = xyz - [0, label_i['h'] / 2, 0]
        xyz_center = np.dot(T_cam_velo, np.append(xyz_center_cam, 1))[:3]
        lwh = np.array([label_i['l'], label_i['w'], label_i['h']])
        semantic_label = label_i['type']

        # Size Heading
        box3d_size = np.array([2 * lwh[0], 2 * lwh[1], 2 * lwh[2]])
        heading_angle = label_i['ry']
        heading_angles.append(heading_angle)

        # # augment
        # if self.training:
        #     rand_roty_angle = (np.random.rand() * 2 - 1.) * 5. / 180 * np.pi
        #     rand_scale = (np.random.rand() * 2 - 1.) * 0.1 + 1.
        #     heading_angle += rand_roty_angle
        #     xyz_center = xyz_center * rand_scale
        #     box3d_size = box3d_size * rand_scale

        size_class, size_residual = size2class(box3d_size, semantic_label)
        heading_class, heading_residual = angle2class(heading_angle, NUM_HEADING_BIN)

        bboxes_xyz.append(xyz_center)
        bboxes_lwh.append(lwh)
        semantic_labels.append(type2class[semantic_label])
        heading_labels.append(heading_class)
        heading_residuals.append(heading_residual)
        size_labels.append(size_class)
        size_residuals.append(size_residual)
        bboxes_trackID.append(label_i['track_id'])

    label_process = {
        'bboxes_trackID': bboxes_trackID,
        'bboxes_num': len(label),
        'bboxes_xyz': bboxes_xyz,
        'bboxes_lwh': bboxes_lwh,
        'semantic_label': semantic_labels,
        'heading_labels': heading_labels,
        'heading_residuals': heading_residuals,
        'size_labels': size_labels,
        'size_residuals': size_residuals,
        'heading_angle': heading_angles,
        'T_cam_velo': T_cam_velo,
        'data_path': datapath
    }

    return label_process


def size2class(size, type_name):
    ''' Convert 3D box size (l,w,h) to size class and size residual '''
    size_class = type2class[type_name]
    size_residual = size - type_mean_size[type_name]
    return size_class, size_residual


def class2size(pred_cls, residual):
    ''' Inverse function to size2class '''
    mean_size = type_mean_size[class2type[pred_cls]]
    return mean_size + residual


def dataset_exam(pointcloud, label):
    print('data_path: ', label['data_path'])
    print('bboxes_num: ', label['bboxes_num'])

    bboxes = []
    for i in range(label['bboxes_num']):
        [l, w, h] = label['bboxes_lwh'][i]
        angle = label['heading_angle'][i]
        bbox_points_cam = bbox(l, w, h, roty_angle=angle)
        bbox_points = transform_points(bbox_points_cam, label['T_cam_velo'])
        offset = label['bboxes_xyz'][i]
        bbox_points = bbox_points + offset
        # bbox_points = bbox_points[:, [1,0,2]]
        bbox_lines = [[0, 1], [0, 3], [1, 2], [2, 3], [4, 5], [4, 7], [5, 6], [6, 7],
                 [0, 4], [1, 5], [2, 6], [3, 7]]
        colors = [[1, 0, 0] for i in range(len(bbox_lines))]
        line_set = o3d.geometry.LineSet()
        line_set.points = o3d.utility.Vector3dVector(bbox_points)
        line_set.lines = o3d.utility.Vector2iVector(bbox_lines)
        line_set.colors = o3d.utility.Vector3dVector(colors)
        bboxes.append(line_set)

    pc_temp = o3d.geometry.PointCloud()
    pc_temp.points = o3d.utility.Vector3dVector(pointcloud)
    # IPython.embed()
    bboxes.append(pc_temp)
    o3d.visualization.draw_geometries(bboxes)


def transform_points(points, Tr):
    return np.dot(Tr, np.append(points, np.ones((points.shape[0], 1)), axis=1).transpose()).transpose()[:, :-1]


def bbox(l, w, h, roty_angle=None):
    x_corners = [l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2]
    y_corners = [h / 2, h / 2, h / 2, h / 2, -h / 2, -h / 2, -h / 2, -h / 2]
    z_corners = [w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2]
    corners_3d = np.vstack([x_corners, y_corners, z_corners])
    if roty_angle:
        corners_3d = roty(roty_angle) @ corners_3d
    return np.transpose(corners_3d)


def visualize_track_gt(pc1, pc2, tracking_label1, bboxes_label1, bboxes_label2):
    pc1_tracked = pc1 + tracking_label1[:, -3:]
    pc_save = np.append(pc1, pc1_tracked, axis=0)
    pc_save = np.append(pc_save, pc2, axis=0)
    color_pc1 = get_color_vec(color='red', point_num=pc1.shape[0])
    color_pc2 = get_color_vec(color='green', point_num=pc2.shape[0])
    color_pc1_tracked = get_color_vec(color='blue', point_num=pc1_tracked.shape[0])
    color_pc_save = np.append(color_pc1, color_pc1_tracked, axis=0)
    color_pc_save = np.append(color_pc_save, color_pc2, axis=0)
    pc_save_o3d = o3d.geometry.PointCloud()
    pc_save_o3d.points = o3d.utility.Vector3dVector(pc_save)
    pc_save_o3d.colors = o3d.utility.Vector3dVector(color_pc_save)
    # o3d.visualization.draw_geometries([pc_save_o3d])

    # draw bboxes
    bboxes = []
    for i in range(bboxes_label1['bboxes_num']):
        [l, w, h] = bboxes_label1['bboxes_lwh'][i]
        angle = bboxes_label1['heading_angle'][i]
        bbox_points_cam = bbox(l, w, h, roty_angle=angle)
        bbox_points = transform_points(bbox_points_cam, bboxes_label1['T_cam_velo'])
        offset = bboxes_label1['bboxes_xyz'][i]
        bbox_points = bbox_points + offset
        # bbox_points = bbox_points[:, [1,0,2]]
        bbox_lines = [[0, 1], [0, 3], [1, 2], [2, 3], [4, 5], [4, 7], [5, 6], [6, 7],
                      [0, 4], [1, 5], [2, 6], [3, 7]]
        colors = [[1, 0, 0] for i in range(len(bbox_lines))]
        line_set = o3d.geometry.LineSet()
        line_set.points = o3d.utility.Vector3dVector(bbox_points)
        line_set.lines = o3d.utility.Vector2iVector(bbox_lines)
        line_set.colors = o3d.utility.Vector3dVector(colors)
        bboxes.append(line_set)
    for i in range(bboxes_label2['bboxes_num']):
        [l, w, h] = bboxes_label2['bboxes_lwh'][i]
        angle = bboxes_label2['heading_angle'][i]
        bbox_points_cam = bbox(l, w, h, roty_angle=angle)
        bbox_points = transform_points(bbox_points_cam, bboxes_label2['T_cam_velo'])
        offset = bboxes_label2['bboxes_xyz'][i]
        bbox_points = bbox_points + offset
        # bbox_points = bbox_points[:, [1,0,2]]
        bbox_lines = [[0, 1], [0, 3], [1, 2], [2, 3], [4, 5], [4, 7], [5, 6], [6, 7],
                      [0, 4], [1, 5], [2, 6], [3, 7]]
        colors = [[1, 0, 0] for i in range(len(bbox_lines))]
        line_set = o3d.geometry.LineSet()
        line_set.points = o3d.utility.Vector3dVector(bbox_points)
        line_set.lines = o3d.utility.Vector2iVector(bbox_lines)
        line_set.colors = o3d.utility.Vector3dVector(colors)
        bboxes.append(line_set)

    # IPython.embed()
    bboxes.append(pc_save_o3d)
    o3d.visualization.draw_geometries(bboxes)


def dataset_object_detection_exam(pointcloud, label, bboxes_label):
    colors = np.zeros((pointcloud.shape[0], 3))
    for i in range(pointcloud.shape[0]):
        if label[i, 0] == 1:
            colors[i, 0] = 1
        elif label[i, 1] == 1:
            colors[i, 2] = 1
        else:
            colors[i, 1] = 1

    pc_temp = o3d.geometry.PointCloud()
    pc_temp.points = o3d.utility.Vector3dVector(pointcloud)
    pc_temp.colors = o3d.utility.Vector3dVector(colors)

    bboxes = []
    for i in range(bboxes_label['bboxes_num']):
        [l, w, h] = bboxes_label['bboxes_lwh'][i]
        angle = bboxes_label['heading_angle'][i]
        bbox_points_cam = bbox(l, w, h, roty_angle=angle)
        bbox_points = transform_points(bbox_points_cam, bboxes_label['T_cam_velo'])
        offset = bboxes_label['bboxes_xyz'][i]
        bbox_points = bbox_points + offset
        # bbox_points = bbox_points[:, [1,0,2]]
        bbox_lines = [[0, 1], [0, 3], [1, 2], [2, 3], [4, 5], [4, 7], [5, 6], [6, 7],
                      [0, 4], [1, 5], [2, 6], [3, 7]]
        colors = [[1, 0, 0] for i in range(len(bbox_lines))]
        line_set = o3d.geometry.LineSet()
        line_set.points = o3d.utility.Vector3dVector(bbox_points)
        line_set.lines = o3d.utility.Vector2iVector(bbox_lines)
        line_set.colors = o3d.utility.Vector3dVector(colors)
        bboxes.append(line_set)
    # IPython.embed()
    # o3d.visualization.draw_geometries([pc_temp])
    bboxes.append(pc_temp)
    o3d.visualization.draw_geometries(bboxes)


if __name__ == '__main__':
    DATA = '/home/skwang/data/KITTI_object_tracking/training'
    NUM_POINT = 20000
    for index in range(0,50):
        # TRAIN_DATASET = LoadDataset(DATA, npoints=NUM_POINT, train=True, shuffle=True)
        TRAIN_DATASET = LoadDataset(DATA, npoints=NUM_POINT, aug_displacement=1.5)
        # point_cloud, tracking_classification_label_f1, classification_label_f2, bboxes_label_f1, bboxes_label_f2 = TRAIN_DATASET[index]
        point_cloud, classification_label_f1, classification_label_f2, tracking_label, bboxes_label_f1, bboxes_label_f2 = TRAIN_DATASET[index]
        tracking_classification_label_f1 = np.append(classification_label_f1, tracking_label[:, :3], axis=-1)
        pc1 = point_cloud[0:NUM_POINT, :]
        pc2 = point_cloud[NUM_POINT:, :]
        visualize_track_gt(pc1, pc2, tracking_classification_label_f1, bboxes_label_f1, bboxes_label_f2)
        # dataset_object_detection_exam(pc1, tracking_classification_label_f1, bboxes_label_f1)
        # dataset_object_detection_exam(pc2, classification_label_f2, bboxes_label_f2)

        # dataset_object_detection_exam(pc1, classification_label_f1, bboxes_label_f1)
        # # idxs = np.random.choice(pc1.shape[0], pc1.shape[0]-1, replace=False)
        # idxs = np.arange(0, pc1.shape[0])
        # np.random.shuffle(idxs)
        #
        # pc_a = pc1[idxs]
        # a = classification_label_f1[idxs]
        #
        # dataset_object_detection_exam(pc_a, a, bboxes_label_f1)
        #
        # pc_all = np.append(pc1, pc_a + np.array([0,0,5]), axis=0)
        # a_all = np.append(classification_label_f1, a, axis=0)
        # dataset_object_detection_exam(pc_all, a_all, bboxes_label_f1)
        # IPython.embed()

        # colors = np.zeros((pc1.shape[0], 3))
        # for i in range(pointcloud.shape[0]):
        #     if label[i, 0] == 1:
        #         colors[i, 0] = 1
        #     else:
        #         colors[i, 2] = 1
        #
        # pc_temp = o3d.geometry.PointCloud()
        # pc_temp.points = o3d.utility.Vector3dVector(pointcloud)
        # pc_temp.colors = o3d.utility.Vector3dVector(colors)
    # point_cloud, label_f1, label_f2 = TRAIN_DATASET[idxs[0]]

    # pass
    # # dataset_viz()
    # # get_box3d_dim_statistics('/home/rqi/Data/mysunrgbd/training/train_data_idx.txt')
    # # extract_roi_seg('/home/rqi/Data/mysunrgbd/training/val_data_idx.txt', 'training',
    # #                 output_filename='val_1002.zip.pickle', viz=False, augmentX=1)
    # # extract_roi_seg('/home/rqi/Data/mysunrgbd/training/train_data_idx.txt', 'training',
    # #                 output_filename='train.pickle', viz=False, augmentX=1)
    # if __name__ == '__main__':
    #     import mayavi.mlab as mlab
    #     import config
    #
    #     sys.path.append(os.path.join(BASE_DIR, '../../mayavi'))
    #     from viz_utils import draw_lidar, draw_gt_boxes3d
    #
    #     median_list = []
    #     dataset = MyDataFlow('/media/neil/DATA/mysunrgbd', 'training')
    #     dataset.reset_state()
    #     # print(type(dataset.input_list[0][0, 0]))
    #     # print(dataset.input_list[0].shape)
    #     # print(dataset.input_list[2].shape)
    #     # input()
    #     for obj in dataset:
    #         for i in range(len(obj[1])):
    #             data = [o[i] for o in obj]
    #             print('Center: ', data[1], 'angle_class: ', data[4], 'angle_res:', data[5], 'size_class: ', data[6],
    #                   'size_residual:', data[7], 'real_size:', type_mean_size[class2type[data[6]]] + data[7])
    #             box3d_from_label = get_3d_box(class2size(data[6], data[7] * type_mean_size[class2type[data[6]]]), class2angle(data[4], data[5] * np.pi / config.NH, config.NH), data[1])
    #             # raw_input()
    #
    #             ## Recover original labels
    #             # rot_angle = dataset.get_center_view_rot_angle(i)
    #             # print dataset.id_list[i]
    #             # print from_prediction_to_label_format(data[2], data[3], data[4], data[5], data[6], rot_angle)
    #
    #             ps = obj[0]
    #             fig = mlab.figure(figure=None, bgcolor=(0.4, 0.4, 0.4), fgcolor=None, engine=None, size=(1000, 500))
    #             mlab.points3d(ps[:, 0], ps[:, 1], ps[:, 2], mode='point', colormap='gnuplot', scale_factor=1,
    #                           figure=fig)
    #             mlab.points3d(0, 0, 0, color=(1, 1, 1), mode='sphere', scale_factor=0.2, figure=fig)
    #             # draw_gt_boxes3d([dataset.get_center_view_box3d(i)], fig)
    #             draw_gt_boxes3d([box3d_from_label], fig, color=(1, 0, 0))
    #             mlab.orientation_axes()
    #             print(ps[0:10, :])
    #             mlab.show()
    # # extract_roi_seg_from_rgb_detection('FPN_384x384', 'training', 'fcn_det_val.zip.pickle', valid_id_list=[int(line.rstrip()) for line in open('/home/rqi/Data/mysunrgbd/training/val_data_idx.txt')], viz=True)