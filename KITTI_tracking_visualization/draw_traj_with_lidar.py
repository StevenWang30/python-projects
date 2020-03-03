import numpy as np
import matplotlib.pyplot as plt
import pickle
from draw_spatiol_temporal_map.utils import get_center_position_Lidar
from utils_for_tracking.KITTI_dataset_utils.utils import load_bin
from utils_for_tracking.tracking_utils.visualization_utils import draw_birdeye_bbox_shc
from utils_for_tracking.KITTI_dataset_utils.kittitrackingdata_second import KittiCalib
from draw_spatiol_temporal_map.utils import get_center_position_Lidar
import os
import IPython

def visualize(seq, frame_id, pc, traj_seq, vis_2d=False, vis_3d=False, save_2d=False, save_3d=False, save_3d_path=None):
    color_rgb = {'g': '#008000', 'r': '#FF0000', 'b': '#0000FF', 'm': '#FF00FF'}
    fig = plt.figure()
    axes = plt.gca()
    axes.set_aspect('equal')
    axes.set_xlim([0, 60])
    axes.set_ylim([-10, 20])

    # dict_f2_num = 0
    data = None
    for i in range(len(traj_seq)):
        if traj_seq[i]['metadata']['image_seq'] == seq and traj_seq[i]['metadata']['image_idx'] == frame_id:
            data = traj_seq[i]
    if data == None:
        print("There is no frame (%d, %d) in trajectory sequence. " % (seq, frame_id))
        return

    plt.scatter(pc[:, 0], pc[:, 1], s=0.03)

    centers = get_center_position_Lidar(data, T_cam_velo)
    for b in range(len(data['bbox'])):
        if data['track_id'][b] < 0:
            continue
        track_id_this_bbox = data['track_id'][b]
        [h, w, l] = data['dimensions'][b]
        yaw = data['rotation_y'][b]
        [x, y, z] = centers[b]
        bboxes_birdeye_points = draw_birdeye_bbox_shc([l, w, h], yaw, [x, y, z], T_cam_velo)
        plt.scatter(bboxes_birdeye_points[:, 0], bboxes_birdeye_points[:, 1], marker='^', c=color_rgb['b'],
                    s=0.005)
        # plt.text(self.data['xyz'][j][0], self.data['xyz'][j][1], str(round(self.data['score'][j], 2)),
        #          ha="center", va="center", color=color_rgb['b'], size=5)
        # dict_f2_num += 1


        # draw trajectory
        traj_point = np.zeros((0, 3))
        for j in range(len(traj_seq)):
            if traj_seq[j]['metadata']['image_seq'] < seq:
                continue
            if traj_seq[j]['metadata']['image_seq'] == seq and traj_seq[j]['metadata']['image_idx'] > frame_id:
                break

            data_i = traj_seq[j]
            centers_i = get_center_position_Lidar(data_i, T_cam_velo)
            for bb_i in range(len(data_i['bbox'])):
                if data_i['track_id'][bb_i] == track_id_this_bbox:
                    traj_point = np.append(traj_point, [centers_i[bb_i]], axis=0)
                    break
        if traj_point.shape[0] > 1:
            # plt.scatter(traj_point[:, 0], traj_point[:, 1], marker='.', c=color_rgb['r'])
            plt.plot(traj_point[:, 0], traj_point[:, 1], linewidth=2)
                # elif traj_idx == 2:
                #     plt.plot(pos_traj[:, 0], pos_traj[:, 1])
                # else:
                #     x = pos_traj[:, 0]
                #     y = pos_traj[:, 1]
                #     xnew = np.linspace(x.min(), x.max(), 20)  # 300 represents number of points to make between x.min and x.max
                #     y_smooth = spline(x, y, xnew)
                #     plt.plot(xnew, y_smooth)


    if vis_2d:
        print("visualize...")
        plt.show()



import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', default='/data/KITTI_object_tracking/testing')
parser.add_argument('--velodyne_dir', default='/data/KITTI_object_tracking/testing/velodyne')
parser.add_argument('--pkl_path', default='/home/skwang/PYProject/KITTI_tracking_visualization/KITTI_test_trajectory.pkl')
parser.add_argument('--seq', default=0)
parser.add_argument('--frame', default=10)
parser.add_argument('--vis', default=True)
parser.add_argument('--save', default=False)
args = parser.parse_args()

seq = int(args.seq)
frame = int(args.frame)
traj_pkl_name = args.pkl_path
traj_data = pickle.load(open(traj_pkl_name, 'rb'))
data_dir = args.data_dir
velodyne_dir = args.velodyne_dir
# pose_dir = '/data/KITTI_object_tracking/testing/pose'

# get calib
calib_path = os.path.join(data_dir, "calib", '%04d.txt' % seq)
calib_seq = KittiCalib(calib_path).read_calib_file()
T_cam_velo = calib_seq.Tr_cam_to_velo

# get points
velo_points = load_bin(os.path.join(velodyne_dir, '%04d' % seq, '%06d.bin' % frame))


visualize(seq, frame, velo_points, traj_data, vis_2d=args.vis, save_2d=args.save)


