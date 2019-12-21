import pickle
import IPython
import argparse
import os
import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from tracking_utils.KITTI_dataset_utils.kittitrackingdata_second import *
from tracking_utils.KITTI_dataset_utils.dataset import transform_points

"""Parse input arguments."""
parser = argparse.ArgumentParser(description='SORT demo')
parser.add_argument('--display', dest='display', help='Display online tracker output (slow) [False]',
                    action='store_true')
parser.add_argument('--output_dir', default='/data/KITTI_object_tracking/spatio-temporal-map/raw_detection_map')

parser.add_argument('--detection_data_pkl',
                    default='/data/KITTI_object_tracking/results_PointRCNNTrackNet/detection_pkl/training_result.pkl')
parser.add_argument('--tracking_predict_pkl',
                    default='/data/KITTI_object_tracking/results_PointRCNNTrackNet/tracking_pkl/training_result.pkl')
parser.add_argument('--tracking_label_pkl',
                    default='/home/skwang/PYProject/draw_spatiol-temporal_map/pkl_data/training_label_result.pkl')

parser.add_argument('--data-dir', dest='data_dir', default='/data/KITTI_object_tracking/training')
parser.add_argument('--method', default='PointRCNN')

parser.add_argument('--map_duration_frame', default=40)

parser.add_argument('--is_test', default=False)
parser.add_argument('--seq_start', default= 0 )
parser.add_argument('--seq_end',   default= 20 )
args = parser.parse_args()

print("******************************************************")
print("*******************Detection Method*******************")
print("*********************", args.method, "**********************")
print("******************************************************\n")

type_whitelist = ('Car', 'Van')

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


def get_center_position_Lidar(det_frame):
    # calib_f1, img_f1, label_f1, pc_f1 = KittiTrackingData(root_dir=args.data_dir,
    #                                                       seq=det_frame['metadata']['image_seq'],
    #                                                       idx=det_frame['metadata']['image_idx'],
    #                                                       is_test=args.is_test).read_data()

    T_cam_velo = calib_seq.Tr_cam_to_velo
    centers = []
    for i in range(len(det_frame['location'])):
        center = transform_points(np.array([det_frame['location'][i]]), T_cam_velo)[0]
        l, h, w = det_frame['dimensions'][i]
        center[2] += h / 2
        centers.append(center)
    return centers


def get_trajectory(tracking_data):
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
                centers = get_center_position_Lidar(tracking_data[i])
                [x, y, z] = centers[pos]
                t = tracking_data[i]['metadata']['image_idx']
                trajectory_i.append([t, x, y])
        if len(trajectory_i) == 0:
            continue
        trajectories.append(trajectory_i)

    return trajectories




dt_data = pickle.load(open(args.detection_data_pkl, 'rb'))
tracking_data = pickle.load(open(args.tracking_predict_pkl, 'rb'))
tracking_label_data = pickle.load(open(args.tracking_label_pkl, 'rb'))

for seq in range(args.seq_start, args.seq_end + 1):
    output_dir = os.path.join(args.output_dir, '%4d' % seq)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # get calib
    calib_path = os.path.join(args.data_dir, "calib", '%04d.txt' % seq)
    calib_seq = KittiCalib(calib_path).read_calib_file()

    # map_num = math.ceil(len(det_seq) / args.map_duration_frame)
    MAX_ = 999999
    for i in range(MAX_):
        start_idx = args.map_duration_frame * i
        end_idx = args.map_duration_frame * (i + 1)
        # get the detection result in this sequence
        det_seq = []
        for i in range(len(dt_data)):
            if dt_data[i]['metadata']['image_seq'] < seq:
                continue
            if dt_data[i]['metadata']['image_seq'] > seq:
                break
            if dt_data[i]['metadata']['image_idx'] < start_idx:
                continue
            if dt_data[i]['metadata']['image_idx'] >= end_idx:
                break
            det_seq.append(dt_data[i])

        if len(det_seq) == 0:
            break

        # get the predict tracking result in this sequence
        tracking_seq = []
        for i in range(len(tracking_data)):
            if tracking_data[i]['metadata']['image_seq'] < seq:
                continue
            if tracking_data[i]['metadata']['image_seq'] > seq:
                break
            if tracking_data[i]['metadata']['image_idx'] < start_idx:
                continue
            if tracking_data[i]['metadata']['image_idx'] >= end_idx:
                break
            tracking_seq.append(tracking_data[i])

        # get the predict tracking result in this sequence
        tracking_label_seq = []
        for i in range(len(tracking_label_data)):
            if tracking_label_data[i]['metadata']['image_seq'] < seq:
                continue
            if tracking_label_data[i]['metadata']['image_seq'] > seq:
                break
            if tracking_label_data[i]['metadata']['image_idx'] < start_idx:
                continue
            if tracking_label_data[i]['metadata']['image_idx'] >= end_idx:
                break
            tracking_label_seq.append(tracking_label_data[i])


        # spatio_temporal_t_x_map_points = []
        # spatio_temporal_t_y_map_points = []
        spatio_temporal_t_x_y_map_points = []
        for j in range(len(det_seq)):
            centers = get_center_position_Lidar(det_seq[j])
            for c in range(len(centers)):
                [x, y, z] = centers[c]
                t = det_seq[j]['metadata']['image_idx']
                # spatio_temporal_t_x_map_points.append([t, x])
                # spatio_temporal_t_y_map_points.append([t, y])
                spatio_temporal_t_x_y_map_points.append([t, x, y])
        # print("spatio_temporal_t_x_map_points: ", spatio_temporal_t_x_map_points)
        print("\nseq ", seq)
        print("frame from ", start_idx, " to ", end_idx)
        # print("visualize the t-x spatio-temporal map... time duration is ", args.map_duration_frame)
        # draw_points(spatio_temporal_t_x_map_points, vis=True)
        # print("visualize the t-y spatio-temporal map... time duration is ", args.map_duration_frame, "\n\n\n")
        # draw_points(spatio_temporal_t_y_map_points, vis=True)

        # print("visualize the t-x-y 3D spatio-temporal map... time duration is ", args.map_duration_frame, "\n\n\n")
        # draw_points(spatio_temporal_t_x_y_map_points, vis=True, v3d=True)

        pred_trajectories = get_trajectory(tracking_seq)
        label_trajectories = get_trajectory(tracking_label_seq)
        draw_3d(spatio_temporal_t_x_y_map_points, pred_trajectories, label_trajectories, vis=True)
