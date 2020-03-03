import pickle
import IPython
import argparse
import os

from draw_spatiol_temporal_map.utils import *


"""Parse input arguments."""
parser = argparse.ArgumentParser(description='SORT demo')
parser.add_argument('--display', dest='display', help='Display online tracker output (slow) [False]',
                    action='store_true')
parser.add_argument('--output_dir', default='/data/KITTI_object_tracking/spatio-temporal-map/raw_detection_map')

parser.add_argument('--tracking_predict_pkl',
                    default='/home/skwang/PYProject/KITTI_tracking_visualization/KITTI_test_trajectory.pkl')
parser.add_argument('--pose_dir',
                    default='/data/KITTI_object_tracking/testing/pose')
parser.add_argument('--velodyne_dir',
                    default='/data/KITTI_object_tracking/testing/velodyne')

parser.add_argument('--data-dir', dest='data_dir', default='/data/KITTI_object_tracking/testing')
parser.add_argument('--method', default='PointRCNN')

parser.add_argument('--frame', default='global', help="frame: -- global, -- inertial")

parser.add_argument('--seq', default= 20 )
args = parser.parse_args()

print("******************************************************")
print("*******************Detection Method*******************")
print("*********************", args.method, "**********************")
print("******************************************************\n")

type_whitelist = ('Car', 'Van')



tracking_data = pickle.load(open(args.tracking_predict_pkl, 'rb'))

seq = int(args.seq)
output_dir = os.path.join(args.output_dir, '%04d' % seq)
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# get calib
calib_path = os.path.join(args.data_dir, "calib", '%04d.txt' % seq)
calib_seq = KittiCalib(calib_path).read_calib_file()
T_cam_velo = calib_seq.Tr_cam_to_velo

# get pose
pose_seq = load_pose(args.velodyne_dir, args.pose_dir, seq)

# map_num = math.ceil(len(det_seq) / args.map_duration_frame)
# start_idx = int(args.map_duration_frame * i)
# end_idx = int(args.map_duration_frame * (i + 1))
start_idx = 0
end_idx = 0
# get the predict tracking result in this sequence
tracking_seq = []
for i in range(len(tracking_data)):
    if tracking_data[i]['metadata']['image_seq'] < seq:
        continue
    if tracking_data[i]['metadata']['image_seq'] > seq:
        break
    if tracking_data[i]['metadata']['image_idx'] < start_idx:
        continue
    tracking_seq.append(tracking_data[i])
    end_idx += 1
    if end_idx > 50:
        break


# spatio_temporal_t_x_map_points = []
# spatio_temporal_t_y_map_points = []
spatio_temporal_t_x_y_map_points = []
for j in range(len(tracking_seq)):
    centers = get_center_position_Lidar(tracking_seq[j], T_cam_velo)
    # ry_vecs = get_rotation_y_vec_Lidar(det_seq[j], T_cam_velo)
    for c in range(len(centers)):
        [x, y, z] = centers[c]
        t = tracking_seq[j]['metadata']['image_idx']

        if args.frame == 'global':
            [x_global, y_global, z_global] = transform_points_from_inertial_to_global([x, y, z], pose_seq[t])
            spatio_temporal_t_x_y_map_points.append([t, x_global, y_global])
            # [ry_vec_x_global, ry_vec_y_global, ry_vec_z_global] = transform_rotation_y_from_inertial_to_global(
            #     ry_vecs[c], pose_seq[t])
            # print("in ", det_seq[j]['metadata'], ", ry_vec: ", [ry_vec_x_global, ry_vec_y_global, ry_vec_z_global])
        else:
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

pred_trajectories = get_trajectory(tracking_seq, T_cam_velo, pose_seq, type_whitelist, frame=args.frame)
draw_3d(spatio_temporal_t_x_y_map_points, pred_trajectories, vis=True)
