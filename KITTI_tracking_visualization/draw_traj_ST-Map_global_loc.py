import pickle
import IPython
import argparse
import os

from draw_spatiol_temporal_map.utils import *


"""Parse input arguments."""
parser = argparse.ArgumentParser()

parser.add_argument('--tracking_pkl', default='/data/Nuscences/gt_tracking_pkl/label_train.pkl')

parser.add_argument('--frame', default='inertial', help="frame: -- global, -- inertial")

args = parser.parse_args()

# type_whitelist = ('Car', 'Van')



tracking_data = pickle.load(open(args.tracking_pkl, 'rb'))

seq_max = 99999
for seq in range(seq_max):
    start_idx = 0
    end_idx = 0
    # get the predict tracking result in this sequence
    tracking_seq = []
    for i in range(len(tracking_data)):
        if tracking_data[i]['metadata']['seq_idx'] < seq:
            continue
        if tracking_data[i]['metadata']['seq_idx'] > seq:
            break
        if tracking_data[i]['metadata']['frame_idx'] < start_idx:
            continue
        tracking_seq.append(tracking_data[i])
        end_idx += 1
        # if end_idx > 10:
        #     break
    if len(tracking_seq) == 0:
        break

    # spatio_temporal_t_x_map_points = []
    # spatio_temporal_t_y_map_points = []
    spatio_temporal_t_x_y_map_points = []
    for j in range(len(tracking_seq)):
        if args.frame == 'global':
            centers = tracking_seq[j]['global_location']
        else:
            centers = tracking_seq[j]['location']
        for c in range(len(centers)):
            [x, y, z] = centers[c]
            t = tracking_seq[j]['metadata']['frame_idx']
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

    pred_trajectories = get_trajectory(tracking_seq, None, None, None, frame=args.frame)
    draw_3d(spatio_temporal_t_x_y_map_points, pred_trajectories, vis=True)
