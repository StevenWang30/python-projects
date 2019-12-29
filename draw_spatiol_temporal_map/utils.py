import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from tracking_utils.KITTI_dataset_utils.kittitrackingdata_second import *
from tracking_utils.KITTI_dataset_utils.dataset import transform_points
import scipy.io as sio
import math
import IPython
import os

def draw_3d(points, pred_trajectories=None, label_trajectories=None, lines=None, save_path=None, save=False, vis=False):
    # points: [frame, x, y]
    # trajectories: [tra_1: [frame, x, y], tra_2: [frame, x, y]...], where frames in each tra are not repeated.
    if not save and not vis:
        return 0
    points = np.array(points)
    fig = plt.figure()
    colors = {'g': '#008000', 'r': '#FF0000', 'b': '#0000FF', 'm': '#FF00FF'}
    ax = plt.axes(projection='3d')

    if points is not None:
        # plot points
        print("have ", len(points), " points.")
        points = np.array(points)
        if not len(points) == 0:
            zdata = points[:, 0]
            xdata = points[:, 1]
            ydata = points[:, 2]
            ax.scatter3D(xdata, ydata, zdata, c=zdata, cmap='Greens')

    if pred_trajectories is not None:
        # plot trajectories
        print("have ", len(pred_trajectories), " predicted trajectories.")
        for i in range(len(pred_trajectories)):
            tra = np.array(pred_trajectories[i])
            if not len(tra) == 0:
                zdata = tra[:, 0]
                xdata = tra[:, 1]
                ydata = tra[:, 2]
                ax.plot3D(xdata, ydata, zdata, 'blue', linewidth=3)

    if label_trajectories is not None:
        print("have ", len(label_trajectories), " label trajectories.")
        for i in range(len(label_trajectories)):
            tra = np.array(label_trajectories[i])
            if not len(tra) == 0:
                zdata = tra[:, 0]
                xdata = tra[:, 1]
                ydata = tra[:, 2]
                ax.plot3D(xdata, ydata, zdata, 'red')

    if lines is not None:
        '''
            quadratic curve equation in x and y direction:
            x = a*z^2 + b*z + c
            y = d*z^2 + e*z + f
            L[i] = [[a, b, c], [d, e, f]]
        '''
        print("have ", len(lines), " fitting curves.")
        max_z = max(points[:, 0])
        min_z = min(points[:, 0])
        z = np.arange(min_z, max_z + 1)
        for i in range(len(lines)):
            x = lines[i][0][0]*z**2 + lines[i][0][1]*z + lines[i][0][2]
            y = lines[i][1][0]*z**2 + lines[i][1][1]*z + lines[i][1][2]
            ax.plot3D(x, y, z, 'yellow')

    if save:
        print("save fig into ", save_path)
        plt.savefig(save_path, dpi=800)
    if vis:
        print("visualize...")
        plt.show()
    if not save and not vis:
        print("do nothing.")
    plt.close('all')
    return 0

def draw_points(points, save_path=None, save=False, vis=False, v3d=False):
    # points: [frame, x or y value]
    if not save and not vis:
        return 0

    points = np.array(points)

    if not v3d:
        fig = plt.figure()
        colors = {'g': '#008000', 'r': '#FF0000', 'b': '#0000FF', 'm': '#FF00FF'}
        if not len(points.shape) == 2:
            print("the points to draw must have two dimensions.")
            raise AssertionError
        if not points.shape[1] == 2:
            print("the points must in xy plane.")
            raise AssertionError

        plt.scatter(points[:, 0], points[:, 1])
        # for i in range(points.shape[0]):
        #     plt.scatter(points[i, 0], points[i, 1], c=colors['r'], s=0.005, alpha=0.5)

    else:
        fig = plt.figure()
        colors = {'g': '#008000', 'r': '#FF0000', 'b': '#0000FF', 'm': '#FF00FF'}
        ax = plt.axes(projection='3d')

        # 三维散点的数据
        points = np.array(points)
        zdata = points[:, 0]
        xdata = points[:, 1]
        ydata = points[:, 2]
        ax.scatter3D(xdata, ydata, zdata, c=zdata, cmap='Greens')

    if save:
        plt.savefig(save_path, dpi=800)
    elif vis:
        plt.show()
    plt.close('all')  # 关闭图 0
    return 0


def get_center_position_Lidar(det_frame, T_cam_velo):
    # calib_f1, img_f1, label_f1, pc_f1 = KittiTrackingData(root_dir=args.data_dir,
    #                                                       seq=det_frame['metadata']['image_seq'],
    #                                                       idx=det_frame['metadata']['image_idx'],
    #                                                       is_test=args.is_test).read_data()
    centers = []
    for i in range(len(det_frame['location'])):
        center = transform_points(np.array([det_frame['location'][i]]), T_cam_velo)[0]
        l, h, w = det_frame['dimensions'][i]
        center[2] += h / 2
        centers.append(center)
    return centers


def get_rotation_y_vec_Lidar(det_frame, T_cam_velo):
    vecs = []
    for i in range(len(det_frame['rotation_y'])):
        ry = det_frame['rotation_y'][i]
        vec = transform_points(np.array([[-math.sin(ry), 0, math.cos(ry)]]), T_cam_velo)[0]
        vecs.append(vec)
    return vecs


def get_trajectory(tracking_data, T_cam_velo, pose_seq, type_whitelist=('Car', 'Van'), frame='global'):
    max_trajectory_idx = 0
    for i in range(len(tracking_data)):
        if len(tracking_data[i]['track_id']) == 0:
            continue
        if max(tracking_data[i]['track_id']) > max_trajectory_idx:
            max_trajectory_idx = max(tracking_data[i]['track_id'])
    max_trajectory_idx = int(max_trajectory_idx) + 1

    trajectories = []
    for t_id in range(max_trajectory_idx):
        trajectory_i = []
        for i in range(len(tracking_data)):
            if t_id in tracking_data[i]['track_id']:
                temp_list = list(tracking_data[i]['track_id'])
                pos = temp_list.index(t_id)
                if not tracking_data[i]['name'][pos] in type_whitelist:
                    continue
                centers = get_center_position_Lidar(tracking_data[i], T_cam_velo)
                [x, y, z] = centers[pos]
                t = tracking_data[i]['metadata']['image_idx']
                if frame == 'global':
                    [x_global, y_global, z_global] = transform_points_from_inertial_to_global([x, y, z], pose_seq[t])
                    trajectory_i.append([t, x_global, y_global])
                else:
                    trajectory_i.append([t, x, y])
        if len(trajectory_i) == 0:
            continue
        trajectories.append(trajectory_i)

    return trajectories



def load_pose(velodyne_dir, pose_dir, seq):
    # from velodyne bin directory read the max pose length.
    # (KITTI seq 1 lost 176-180 frame bin file. so it is important.
    dir_path = os.path.join(velodyne_dir, "%04d" % seq)
    bin_files = os.listdir(dir_path)
    bin_files.sort(key=lambda x: int(x.split('.')[0]))
    data_num = int(bin_files[-1].split('.')[0]) + 1
    data = np.zeros((data_num, 4, 4))
    data[0] = np.asarray([[1.0, 0.0, 0.0, 0.0],
                         [0.0, 1.0, 0.0, 0.0],
                         [0.0, 0.0, 1.0, 0.0],
                         [0.0, 0.0, 0.0, 1.0]])

    pose_path = os.path.join(pose_dir, '%04d.mat' % seq)
    mat_data = sio.loadmat(pose_path)
    raw_data = mat_data['pose']
    raw_data = np.einsum('kli->ikl', raw_data)

    null_num = 0
    for t in range(data_num - 1):
        i = t + 1
        if not os.path.exists(os.path.join(dir_path, "%06d.bin" % i)):
            data[i] = data[i - 1]
            null_num += 1
            # print("null_num + 1:, ", null_num)
            continue
        data[i] = np.dot(np.append(raw_data[i - 1 - null_num], [[0, 0, 0, 1]], axis=0), data[i - 1])
        # print("vel_path:", os.path.join(dir_path, "%06d.bin" % i), "data_i: ", i, " raw_data_i: ", i - 1 - null_num, " * data_i-1: ", i - 1)

    print("null_num is : ", null_num)
    return data


def transform_points_from_inertial_to_global(point, T_inertial_global):
    point_4 = np.array(point + [1])
    point_global = np.dot(np.linalg.inv(T_inertial_global), point_4)
    return point_global[:3]

def transform_rotation_y_from_inertial_to_global(ry_vec, T_inertial_global):
    vec_cam = np.array(list(ry_vec) + [1])
    vec_global = np.dot(np.linalg.inv(T_inertial_global), vec_cam)
    scale = math.sqrt(vec_global[0] * vec_global[0] + vec_global[1] * vec_global[1] + vec_global[2] * vec_global[2])
    return vec_global[:3] / scale
