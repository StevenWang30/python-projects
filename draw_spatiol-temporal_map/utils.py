import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from tracking_utils.KITTI_dataset_utils.kittitrackingdata_second import *
from tracking_utils.KITTI_dataset_utils.dataset import transform_points
import scipy.io as sio

def draw_3d(points, pred_trajectories, label_trajectories, save_path=None, save=False, vis=False):
    # points: [frame, x, y]
    # trajectories: [tra_1: [frame, x, y], tra_2: [frame, x, y]...], where frames in each tra are not repeated.
    if not save and not vis:
        return 0
    points = np.array(points)
    fig = plt.figure()
    colors = {'g': '#008000', 'r': '#FF0000', 'b': '#0000FF', 'm': '#FF00FF'}
    ax = plt.axes(projection='3d')

    # plot points
    points = np.array(points)
    if not len(points) == 0:
        zdata = points[:, 0]
        xdata = points[:, 1]
        ydata = points[:, 2]
        ax.scatter3D(xdata, ydata, zdata, c=zdata, cmap='Greens')

    # plot trajectories
    for i in range(len(pred_trajectories)):
        tra = np.array(pred_trajectories[i])
        if not len(tra) == 0:
            zdata = tra[:, 0]
            xdata = tra[:, 1]
            ydata = tra[:, 2]
            ax.plot3D(xdata, ydata, zdata, 'blue')

    for i in range(len(label_trajectories)):
        tra = np.array(label_trajectories[i])
        if not len(tra) == 0:
            zdata = tra[:, 0]
            xdata = tra[:, 1]
            ydata = tra[:, 2]
            ax.plot3D(xdata, ydata, zdata, 'red')

    if save:
        plt.savefig(save_path, dpi=800)
    if vis:
        plt.show()
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


def get_trajectory(tracking_data, T_cam_velo, pose_seq, type_whitelist=('Car', 'Van'), frame='global'):
    max_trajectory_idx = 0
    for i in range(len(tracking_data)):
        if len(tracking_data[i]['track_id']) == 0:
            continue
        if max(tracking_data[i]['track_id']) > max_trajectory_idx:
            max_trajectory_idx = max(tracking_data[i]['track_id'])
    max_trajectory_idx = int(max_trajectory_idx)

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



def load_pose(path):
    mat_data = sio.loadmat(path)
    raw_data = mat_data['pose']
    raw_data = np.einsum('kli->ikl', raw_data)
    data = np.zeros((raw_data.shape[0] + 1, 4, 4))
    data[0] = np.asarray([[1.0, 0.0, 0.0, 0.0],
                         [0.0, 1.0, 0.0, 0.0],
                         [0.0, 0.0, 1.0, 0.0],
                         [0.0, 0.0, 0.0, 1.0]])
    for i in range(raw_data.shape[0]):
        data[i + 1] = np.dot(np.append(raw_data[i], [[0, 0, 0, 1]], axis=0), data[i])
    return data


def transform_points_from_inertial_to_global(point, T_inertial_global):
    point_4 = np.array(point + [1])
    point_global = np.dot(np.linalg.inv(T_inertial_global), point_4)
    return point_global[:3]
