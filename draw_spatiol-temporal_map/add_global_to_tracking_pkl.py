import pickle
import IPython
import argparse
import os
import copy

from utils import *

"""Parse input arguments."""
parser = argparse.ArgumentParser(description='add global')
parser.add_argument('--detection_data_pkl',
                    default='/data/KITTI_object_tracking/results_PointRCNNTrackNet/detection_pkl/training_result.pkl')
parser.add_argument('--tracking_predict_pkl',
                    default='/data/KITTI_object_tracking/results_PointRCNNTrackNet/tracking_pkl/training_result.pkl')
parser.add_argument('--tracking_label_pkl',
                    default='/home/skwang/PYProject/draw_spatiol-temporal_map/pkl_data/training_label_result.pkl')
parser.add_argument('--pose_dir',
                    default='/data/KITTI_object_tracking/training/pose')
parser.add_argument('--velodyne_dir',
                    default='/data/KITTI_object_tracking/training/velodyne')

parser.add_argument('--data-dir', dest='data_dir', default='/data/KITTI_object_tracking/training')
parser.add_argument('--is_test', default=False)
args = parser.parse_args()

type_whitelist = ('Car', 'Van')

# tracking_data = pickle.load(open(args.tracking_predict_pkl, 'rb'))
# tracking_label_data = pickle.load(open(args.tracking_label_pkl, 'rb'))

# process what data
process_data_path = args.tracking_predict_pkl
# process_data_path = args.tracking_label_pkl
process_data = pickle.load(open(process_data_path, 'rb'))
global_data = copy.deepcopy(process_data)

max_seq = int(process_data[-1]['metadata']['image_seq'])
for seq in range(max_seq + 1):
    print("process seq ", seq)
    # get calib
    calib_path = os.path.join(args.data_dir, "calib", '%04d.txt' % seq)
    calib_seq = KittiCalib(calib_path).read_calib_file()
    T_cam_velo = calib_seq.Tr_cam_to_velo

    # get pose
    pose_seq = load_pose(args.velodyne_dir, args.pose_dir, seq)

    # map_num = math.ceil(len(det_seq) / args.map_duration_frame)
    MAX_ = 999999
    # get the data in this sequence
    idx_seq = []
    for i in range(len(process_data)):
        if process_data[i]['metadata']['image_seq'] < seq:
            continue
        if process_data[i]['metadata']['image_seq'] > seq:
            break
        idx_seq.append(i)

    if len(idx_seq) == 0:
        continue

    for j in range(len(idx_seq)):
        print(global_data[idx_seq[j]]['metadata'])
        global_data[idx_seq[j]]['global_location'] = []
        global_data[idx_seq[j]]['global_ry_vec'] = []
        data_i = process_data[idx_seq[j]]
        centers = get_center_position_Lidar(data_i, T_cam_velo)
        ry_vecs = get_rotation_y_vec_Lidar(data_i, T_cam_velo)
        for c in range(len(centers)):
            [x, y, z] = centers[c]
            t = data_i['metadata']['image_idx']
            [x_global, y_global, z_global] = transform_points_from_inertial_to_global([x, y, z], pose_seq[t])
            global_data[idx_seq[j]]['global_location'].append([x_global, y_global, z_global])

            [ry_vec_x_global, ry_vec_y_global, ry_vec_z_global] = transform_rotation_y_from_inertial_to_global(ry_vecs[c], pose_seq[t])
            global_data[idx_seq[j]]['global_ry_vec'].append([ry_vec_x_global, ry_vec_y_global, ry_vec_z_global])

        global_data[idx_seq[j]]['global_location'] = np.array(global_data[idx_seq[j]]['global_location'])
        global_data[idx_seq[j]]['global_ry_vec'] = np.array(global_data[idx_seq[j]]['global_ry_vec'])

save_path = process_data_path.replace('.pkl', '_with_global.pkl')
print("save data into ", save_path)
with open(save_path, 'wb') as f:
    pickle.dump(global_data, f)