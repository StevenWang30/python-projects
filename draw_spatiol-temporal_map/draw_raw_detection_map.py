import pickle
import IPython
import argparse
import os
import math

from utils import *

"""Parse input arguments."""
parser = argparse.ArgumentParser(description='SORT demo')
parser.add_argument('--display', dest='display', help='Display online tracker output (slow) [False]',
                    action='store_true')
parser.add_argument('--output_dir', default='/data/KITTI_object_tracking/spatio-temporal-map/raw_detection_map')
parser.add_argument('--detection_data_pkl',
                    default='/data/KITTI_object_tracking/results_PointRCNNTrackNet/detection_pkl/training_result.pkl')
parser.add_argument('--data-dir', dest='data_dir', default='/data/KITTI_object_tracking/training')

parser.add_argument('--map_duration_frame', default=20)

parser.add_argument('--is_test', default=False)
parser.add_argument('--seq_start', default= 0 )
parser.add_argument('--seq_end',   default= 20 )
args = parser.parse_args()


dt_data = pickle.load(open(args.detection_data_pkl, 'rb'))
for seq in range(args.seq_start, args.seq_end + 1):
    output_dir = os.path.join(args.output_dir, '%4d' % seq)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # get calib
    calib_path = os.path.join(args.data_dir, "calib", '%04d.txt' % seq)
    calib_seq = KittiCalib(calib_path).read_calib_file()
    T_cam_velo = calib_seq.Tr_cam_to_velo

    # get the detection result in this sequence
    det_seq = []
    for i in range(len(dt_data)):
        if dt_data[i]['metadata']['image_seq'] < seq:
            continue
        if dt_data[i]['metadata']['image_seq'] > seq:
            break
        det_seq.append(dt_data[i])

    map_num = math.ceil(len(det_seq) / args.map_duration_frame)
    for i in range(map_num):
        start_idx = args.map_duration_frame * i
        end_idx = min(args.map_duration_frame * (i + 1), len(det_seq))
        spatio_temporal_t_x_map_points = []
        spatio_temporal_t_y_map_points = []
        spatio_temporal_t_x_y_map_points = []
        for j in range(len(det_seq)):
            if det_seq[j]['metadata']['image_idx'] < start_idx:
                continue
            if det_seq[j]['metadata']['image_idx'] >= end_idx:
                break
            t = j - start_idx
            centers = get_center_position_Lidar(det_seq[j], T_cam_velo)
            for c in range(len(centers)):
                [x, y, z] = centers[c]
                spatio_temporal_t_x_map_points.append([t, x])
                spatio_temporal_t_y_map_points.append([t, y])
                spatio_temporal_t_x_y_map_points.append([t, x, y])
        print("spatio_temporal_t_x_map_points: ", spatio_temporal_t_x_map_points)
        print("\nframe from ", start_idx, " to ", end_idx)
        # print("visualize the t-x spatio-temporal map... time duration is ", args.map_duration_frame)
        # draw_points(spatio_temporal_t_x_map_points, vis=True)
        # print("visualize the t-y spatio-temporal map... time duration is ", args.map_duration_frame, "\n\n\n")
        # draw_points(spatio_temporal_t_y_map_points, vis=True)
        print("visualize the t-x-y 3D spatio-temporal map... time duration is ", args.map_duration_frame, "\n\n\n")
        draw_points(spatio_temporal_t_x_y_map_points, vis=True, v3d=True)

