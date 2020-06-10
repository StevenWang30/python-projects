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

parser.add_argument('-k', '--kitti_flag', action='store_true')
# parser.add_argument('--tracking_predict_pkl',
#                     default='/data/KITTI_object_tracking/spatio-temporal-map/tracking_pred/temp_result/detection_with_id_seq17.pkl')
# parser.add_argument('--tracking_predict_pkl',
#                     default='/home/skwang/PYProject/KITTI_tracking_visualization/KITTI_train_trajectory.pkl')
parser.add_argument('--pose_dir',
                    default='/data/KITTI_object_tracking/testing/pose')
parser.add_argument('--velodyne_dir',
                    default='/data/KITTI_object_tracking/testing/velodyne')
parser.add_argument('--data-dir', dest='data_dir', default='/data/KITTI_object_tracking/testing')
parser.add_argument('--method', default='PointRCNN')

parser.add_argument('-c', '--carla_flag', action='store_true')

parser.add_argument('-n', '--nuscences_flag', action='store_true')
# parser.add_argument('--tracking_predict_pkl', default='/data/KITTI_object_tracking/spatio-temporal-map/tracking_pred/temp_result/detection_with_id_seq3.pkl')

parser.add_argument('-f', '--fushikang_flag', action='store_true')
parser.add_argument('--tracking_predict_pkl',
                    default='/data/KITTI_object_tracking/spatio-temporal-map/tracking_pred/temp_result/detection_with_id_seq0.pkl')

parser.add_argument('--frame', default='inertial', help="frame: -- global, -- inertial")

parser.add_argument('--seq', default= 0 )

parser.add_argument('--threshold', default= 0.0, type=float, help='default detection threshold, to filter the detections.' )

args = parser.parse_args()

CARLA = args.carla_flag
KITTI = args.kitti_flag
NUSCENES = args.nuscences_flag
FUSHIKANG = args.fushikang_flag
frame = args.frame


type_whitelist = ('Car', 'Van', 'car', 'pedestrian', 'bicycle')
if FUSHIKANG:
    type_whitelist = ['PEDESTRIAN', 'BICYCLE', 'MOTORBIKE', 'GOLF CAR', 'TRUCK', 'MOTORCYCLIST', 'CAR', 'FORKLIFT', 'CYCLIST']


# if NUSCENES or FUSHIKANG:
#     frame_idx_name = 'frame_idx'
#     seq_idx_name = 'seq_idx'
# else:
#     frame_idx_name = 'image_idx'
#     seq_idx_name = 'image_seq'

frame_idx_name = 'frame_idx'
seq_idx_name = 'seq_idx'

# should and only should one
assert int(KITTI) + int(CARLA) + int(NUSCENES) + int(FUSHIKANG) == 1



seq = int(args.seq)
# if NUSCENES:
args.tracking_predict_pkl = args.tracking_predict_pkl.replace(args.tracking_predict_pkl.split('/')[-1].replace('detection_with_id_seq', '').replace('.pkl', ''), str(seq))

tracking_data = pickle.load(open(args.tracking_predict_pkl, 'rb'))

try:
    test = tracking_data[0]['metadata'][seq_idx_name]
except:
    frame_idx_name = 'image_idx'
    seq_idx_name = 'image_seq'

output_dir = os.path.join(args.output_dir, '%04d' % seq)
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

if KITTI:
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
    if tracking_data[i]['metadata'][seq_idx_name] < seq:
        continue
    if tracking_data[i]['metadata'][seq_idx_name] > seq:
        break
    if tracking_data[i]['metadata'][frame_idx_name] < start_idx:
        continue
    tracking_seq.append(tracking_data[i])
    end_idx += 1
    # if end_idx > 50:
    #     break


# spatio_temporal_t_x_map_points = []
# spatio_temporal_t_y_map_points = []
spatio_temporal_t_x_y_map_points = []
for j in range(len(tracking_seq)):
    t = tracking_seq[j]['metadata'][frame_idx_name]

    if KITTI:
        centers = get_center_position_Lidar(tracking_seq[j], T_cam_velo)

        if frame == 'global':
            centers = transform_points_from_inertial_to_global(centers, pose_seq[t])
        # print('i: ', i, ' and image_idx: ', )
    else:
        if frame == 'global':
            centers = tracking_seq[j]['global_location']
        else:
            centers = tracking_seq[j]['location']

    valid_num = 0
    for c in range(len(centers)):
        if tracking_seq[j]['score'][c] >= args.threshold:
            valid_num += 1
            [x, y, z] = centers[c]
            spatio_temporal_t_x_y_map_points.append([t, x, y])
    print('t: ', t, ', valid detections num: ', valid_num)

# print("spatio_temporal_t_x_map_points: ", spatio_temporal_t_x_map_points)
print("\nseq ", seq)
print("frame from ", start_idx, " to ", end_idx)
# print("visualize the t-x spatio-temporal map... time duration is ", args.map_duration_frame)
# draw_points(spatio_temporal_t_x_map_points, vis=True)
# print("visualize the t-y spatio-temporal map... time duration is ", args.map_duration_frame, "\n\n\n")
# draw_points(spatio_temporal_t_y_map_points, vis=True)

# print("visualize the t-x-y 3D spatio-temporal map... time duration is ", args.map_duration_frame, "\n\n\n")
# draw_points(spatio_temporal_t_x_y_map_points, vis=True, v3d=True)
if KITTI:
    pred_trajectories = get_trajectory(tracking_seq, T_cam_velo, pose_seq, type_whitelist, frame=args.frame)
else:
    pred_trajectories = get_nuscenes_trajectory(tracking_seq, type_whitelist, frame=frame)
draw_3d(spatio_temporal_t_x_y_map_points, pred_trajectories, vis=True)
