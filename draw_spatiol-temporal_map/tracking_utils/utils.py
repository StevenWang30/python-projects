import math
import numpy as np
import os
import sys
import IPython
import shutil


def delete_all_dir_file(path):
    if not os.path.isdir(path):
        os.makedirs(path)
        print("Make dir to " + path)
    else:
        filelist = os.listdir(path)  # 列出该目录下的所有文件名
        for f in filelist:
            filepath = os.path.join(path, f)  # 将文件名映射成绝对路劲
            if os.path.isfile(filepath):  # 判断该文件是否为文件或者文件夹
                os.remove(filepath)  # 若为文件，则直接删除
                print(str(filepath) + " removed!")
            # else:
            #     # 若为folder
            #     shutil.rmtree(filepath)
            #     print(str(filepath) + " removed!")
        print("remove all old files in " + path)


def delete_all_folder(path):
    if not os.path.isdir(path):
        os.makedirs(path)
        print("Make dir to " + path)
    else:
        filelist = os.listdir(path)  # 列出该目录下的所有文件名
        for f in filelist:
            filepath = os.path.join(path, f)  # 将文件名映射成绝对路劲
            # if os.path.isfile(filepath):  # 判断该文件是否为文件或者文件夹
            #     os.remove(filepath)  # 若为文件，则直接删除
            #     print(str(filepath) + " removed!")
            if not os.path.isfile(filepath):
                # 若为folder
                shutil.rmtree(filepath)
                print(str(filepath) + " removed!")
        print("remove all old files in " + path)

def extract_ground_points_from_plane_file(pointcloud, planefile, threshold):
    [a, b, c, d] = np.loadtxt(planefile)
    plane = np.array([a, b, c, d])
    pointcloud = np.array(pointcloud)
    #
    # ground = []
    # deground = []
    # for i in range(pointcloud.shape[0]):
    #     x = pointcloud[i][0]
    #     y = pointcloud[i][1]
    #     z = pointcloud[i][2]
    #     dis = abs(a * x + b * y + c * z + d) / math.sqrt(a * a + b * b + c * c)
    #     if dis < threshold:
    #         ground.append([x, y, z])
    #     else:
    #         deground.append([x, y, z])
    pc = np.append(pointcloud, np.ones((pointcloud.shape[0], 1)), axis=1)
    dis_matrix = np.dot(pc, plane) / math.sqrt(a * a + b * b + c * c)
    # deground_idx = np.where(dis_matrix > threshold)
    # deground_points = pointcloud[deground_idx]

    ground_idx = np.where(dis_matrix <= threshold)
    ground_points = pointcloud[ground_idx]

    return ground_points, ground_idx


def find_points_idx_in_bbox(pointcloud, bbox_center, bbox_size, bbox_angle, T_cam_velo, threshold=0, cover_ground=False):
    # if cover_ground=False --> idx_y has - 0.5 i
    THRESHOLD = np.array([0.6, 0.3, 0.7]) + threshold
    [l, w, h] = bbox_size + THRESHOLD
    pointcloud_dif = pointcloud - bbox_center
    pointcloud_trans = np.transpose(np.dot(roty4(-1 * bbox_angle), np.dot(np.linalg.inv(T_cam_velo),
                   np.transpose(np.append(pointcloud_dif, np.ones((pointcloud_dif.shape[0], 1)), axis=1)))))[:, :3]
    idx_x = np.logical_and(pointcloud_trans[:, 0] <= l / 2.0, pointcloud_trans[:, 0] >= -l / 2.0)
    idx_y = np.logical_and(pointcloud_trans[:, 1] <= w / 2.0 - 0.5, pointcloud_trans[:, 1] >= -w / 2.0)
    idx_z = np.logical_and(pointcloud_trans[:, 2] <= h / 2.0, pointcloud_trans[:, 2] >= -h / 2.0)
    idx = np.logical_and(idx_x, np.logical_and(idx_y, idx_z))
    return idx


def rotx(t):
    """Rotation about the x-axis."""
    c = np.cos(t)
    s = np.sin(t)
    return np.array([[1, 0, 0],
                     [0, c, -s],
                     [0, s, c]])


def roty(t):
    """Rotation about the y-axis."""
    c = np.cos(t)
    s = np.sin(t)
    return np.array([[c, 0, s],
                     [0, 1, 0],
                     [-s, 0, c]])


def roty4(t):
    """Rotation about the y-axis."""
    c = np.cos(t)
    s = np.sin(t)
    return np.array([[c, 0, s, 0],
                     [0, 1, 0, 0],
                     [-s, 0, c, 0],
                     [0, 0, 0, 1]])


def rotz(t):
    """Rotation about the z-axis."""
    c = np.cos(t)
    s = np.sin(t)
    return np.array([[c, -s, 0],
                     [s, c, 0],
                     [0, 0, 1]])


def filter_camera_angle(points):
    """
       Filter camera angles (45 degrees) for KiTTI Datasets
       inputs:
           points (np.array): [#points, >=3]
               orders: [x,y,z]
           T_cam_velo: transformation matrix from cam to lidar
       return:
           pts in the camera angle (45degrees) (lidar frame)
    """
    bool_in = np.logical_and((points[:, 1] < points[:, 0]), (-points[:, 1] < points[:, 0]))

    return points[bool_in]


def angle2class(angle, num_class):
    ''' Convert continuous angle to discrete class
        [optinal] also small regression number from
        class center angle to current angle.

        angle is from 0-2pi (or -pi~pi), class center at 0, 1*(2pi/N), 2*(2pi/N) ...  (N-1)*(2pi/N)
        return is class of int32 of 0,1,...,N-1 and a number such that
            class*(2pi/N) + number = angle
    '''
    angle = angle % (2 * np.pi)
    assert (angle >= 0 and angle <= 2 * np.pi)
    angle_per_class = 2 * np.pi / float(num_class)
    shifted_angle = (angle + angle_per_class / 2) % (2 * np.pi)
    class_id = int(shifted_angle / angle_per_class)
    residual_angle = shifted_angle - (class_id * angle_per_class + angle_per_class / 2)
    return class_id, residual_angle


def class2angle(pred_cls, residual, num_class, to_label_format=True):
    ''' Inverse function to angle2class '''
    angle_per_class = 2 * np.pi / float(num_class)
    angle_center = pred_cls * angle_per_class
    angle = angle_center + residual
    if to_label_format and angle > np.pi:
        angle = angle - 2 * np.pi
    return angle


def get_calibration(calibfile):
    ''' Read in a calibration file and parse into a dictionary.
            Ref: https://github.com/utiasSTARS/pykitti/blob/master/pykitti/utils.py
            '''
    data = {}
    with open(calibfile, 'r') as f:
        for line in f.readlines():
            line = line.rstrip()
            if len(line) == 0: continue

            try:
                key, value = line.split(':', 1)
                try:
                    data[key] = np.array([float(x) for x in value.split()])
                except ValueError:
                    pass
            except ValueError:
                key = line.split(" ")[0]
                value = line.split(" ")[1:]
                try:
                    data[key] = np.array([float(x) for x in value])
                except ValueError:
                    pass
            # The only non-float values in these files are dates, which
            # we don't care about anyway

    T_velo_cam = data['Tr_velo_cam'].reshape((3, 4))
    R_rect = data['R_rect'].reshape((3, 3))
    T_velo_cam = np.dot(R_rect, T_velo_cam)
    T_velo_cam = np.append(T_velo_cam, [[0, 0, 0, 1]], axis=0)
    T_cam_velo = np.linalg.inv(T_velo_cam)
    return data, T_cam_velo


def load_bin(file_name):
    scan = np.fromfile(file_name, dtype=np.float32)
    scan = scan.reshape((-1, 4))
    point_cloud_array = scan[:, :3]
    # pc = o3d.geometry.PointCloud()
    # pc.points = o3d.utility.Vector3dVector(point_cloud_array)
    return point_cloud_array


def load_kitti_tracking_label(self, label_filename, frame):
    lines = [line.rstrip() for line in open(label_filename)]
    label = []
    for i in range(len(lines)):
        frame_i = int(lines[i].split(' ')[0])
        label_i = read_line(lines[i])
        if frame_i == frame and label_i['type'] in self.type_whitelist:
            label.append(label_i)
    return label


def get_color_vec(color='green', point_num=0):
    color_dict = {'green': 1,
                  'blue': 2,
                  'red': 3}
    color_arr = np.zeros((point_num, 3))
    if color_dict[color] == 1:
        color_arr[:, 1] = 1
    elif color_dict[color] == 2:
        color_arr[:, 2] = 1
    else:
        color_arr[:, 0] = 1

    return color_arr


def read_line(line):
    data = line.split(' ')
    track_id = float(data[1])
    data[0] = data[2] # type
    data[1:] = [float(x) for x in data[3:]]

    label = {
                'track_id': track_id,
                # extract label, truncation, occlusion
                'type': data[0],  # 'Car', 'Pedestrian', ...
                'truncation': data[1],  # truncated pixel ratio [0..1]
                'occlusion': int(data[2]),  # 0=visible, 1=partly occluded, 2=fully occluded, 3=unknown
                'alpha': data[3],  # object observation angle [-pi..pi]

                # extract 2d bounding box in 0-based coordinates
                'xmin': data[4],  # left
                'ymin': data[5],  # top
                'xmax': data[6],  # right
                'ymax': data[7],  # bottom
                'box2d': np.array([data[4], data[5], data[6], data[7]]),

                # extract 3d bounding box information
                'h': data[8],  # box height
                'w': data[9],  # box width
                'l': data[10],  # box length (in meters)
                't': (data[11], data[12], data[13]),  # location (x,y,z) in camera coord.
                'ry': data[14]  # yaw angle (around Y-axis in camera coordinates) [-pi..pi]
            }
    return label


def sigmoid(inx):
    return 1 / (1 + np.exp(-inx))


if __name__ == '__main__':
    DATA = '/home/skwang/data/KITTI_object_tracking/training'

    sequence = fn[0].split('/')[-2]
    labelfile = os.path.join(self.root, "label_02", sequence + ".txt")
