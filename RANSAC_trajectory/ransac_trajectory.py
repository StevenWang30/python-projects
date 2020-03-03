# sudo python3 ransac_trajectory.py
import pickle
import IPython
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import argparse
import os
import copy
from draw_spatiol_temporal_map.utils import *
# from utils.quadrics import *
# from utils.least_square import *
from utils.least_square_x_y import *

"""Parse input arguments."""
parser = argparse.ArgumentParser(description='RANSAC trajectory')
parser.add_argument('--detection_data_pkl',
                    default='/data/KITTI_object_tracking/results_PointRCNNTrackNet/detection_pkl/training_result_with_global.pkl')
# parser.add_argument('--tracking_predict_pkl',
#                     default='/data/KITTI_object_tracking/results_PointRCNNTrackNet/tracking_pkl/training_result_with_global.pkl')
# parser.add_argument('--tracking_label_pkl',
#                     default='/home/skwang/PYProject/draw_spatiol-temporal_map/pkl_data/training_label_result_with_global.pkl')
parser.add_argument('--pose_dir',
                    default='/data/KITTI_object_tracking/training/pose')
parser.add_argument('--velodyne_dir',
                    default='/data/KITTI_object_tracking/training/velodyne')

parser.add_argument('--data_dir', dest='data_dir', default='/data/KITTI_object_tracking/training')
parser.add_argument('--method', default='PointRCNN')

parser.add_argument('--RANSAC_threshold', default=1.0)
parser.add_argument('--p_num', default=3)
parser.add_argument('--iterations', default=100)

parser.add_argument('--frame', default='global', help="frame: -- global, -- inertial")

parser.add_argument('--window_size', default=10)
parser.add_argument('--step_size', default=4)

parser.add_argument('--is_test', default=False)
parser.add_argument('--seq_start', default= 0 )
parser.add_argument('--seq_end',   default= 20 )

parser.add_argument('-ts', '--tracking_label_save_flag', action='store_true')
parser.add_argument('--tracking_label_save_dir', default='/data/KITTI_object_tracking/results_RANSAC/tracking_pred')
args = parser.parse_args()

print("******************************************************")
print("*******************Detection Method*******************")
print("*********************", args.method, "**********************")
print("******************************************************\n")

type_whitelist = ('Car', 'Van')

dt_data = pickle.load(open(args.detection_data_pkl, 'rb'))
# tracking_data = pickle.load(open(args.tracking_predict_pkl, 'rb'))
# tracking_label_data = pickle.load(open(args.tracking_label_pkl, 'rb'))

args.iterations = int(args.iterations)
args.seq_start = int(args.seq_start)
args.seq_end = int(args.seq_end)


def ransac_find_trajectory(traj_seq, start_idx, l, det_window, points, p_num, iterations, threshold=1.0, vis=True):
    '''
    :param traj_seq:
        Add 'track_id' into the dict, which save the whole trajectories data as pkl.
    :param start_idx:
        det_seq_idx = start_idx + det_window_idx.
    :param l:
        total number of trajectories.
    :param det_window:
        Deepcopy this to generate trajectories in this window, for visualization. (add 'track_id' into the dict.)
    :param points:
        spatio_temporal_t_x_y_map_points, N * 3, [t, x, y]
    :param iterations:
        use for ransac.
    :param p_num:
        each iteration, sample p_num points for generating the trajectory.
    :return:
        trajectories.
    '''
    print("compose window from ", start_idx)
    origin_points = copy.deepcopy(points)
    tracking_seq = copy.deepcopy(det_window)
    for i in range(len(tracking_seq)):
        tracking_seq[i]['track_id'] = np.ones_like(tracking_seq[i]['score']) * -1

    lines = []
    for sample_idx in range(iterations):
        if len(points) < p_num:
            break
        frame_list = []
        choose_list = []
        for t in range(len(points)):
            if not points[t][0] in frame_list:
                frame_list.append(points[t][0])
        if len(frame_list) > p_num:
            for te in range(3):
                frame_idx = np.random.choice(len(frame_list), p_num, replace=False)
                # make sure frame idx interval >= 2
                # make sure frame idx interval >= 2
                # make sure frame idx interval >= 2
                choose_list = list(np.array(frame_list)[frame_idx])
                choose_list.sort()
                min_interval = min(np.array(choose_list[1:]) - np.array(choose_list[:-1]))
                if min_interval < 2:
                    continue
                break
            # if min_interval < 2:
            #     print("min_interval < 2, chooselist:", choose_list)
        else:
            break
        frame_list = choose_list

        points_sampled = []
        for f in range(len(frame_list)):
            frame = frame_list[f]
            frame_points = []
            for i in range(len(points)):
                if points[i][0] == frame:
                    frame_points.append(points[i])
            choice = np.random.choice(len(frame_points), 1, replace=False)
            points_sampled.append(frame_points[choice[0]])

        L = least_square_x_y(points_sampled)
        # print("L: ", L)
        # print("sampled points:", points_sampled)
        # test_least_square_x_y(points_sampled, L)
        # print("test origin points:")
        # test_least_square_x_y(origin_points, L)

        # loop for best curve,
        # each loop, add the new nodes into this curve, and re-fitting.
        # until no nodes can be included.
        while 1:
            last_nodes_num = len(points_sampled)
            for i in range(len(tracking_seq)):
                if tracking_seq[i]['metadata']['image_idx'] > start_idx + args.window_size:
                    break
                min_idx, min_dis = evaluate_distance_x_y(tracking_seq[i], L, threshold=100, frame=args.frame)
                if min_idx >= 0:
                    if args.frame == 'global':
                        points_this = [tracking_seq[i]['metadata']['image_idx'],
                                       tracking_seq[i]['global_location'][min_idx][0],
                                       tracking_seq[i]['global_location'][min_idx][1]]
                    else:
                        points_this = [tracking_seq[i]['metadata']['image_idx'],
                                       tracking_seq[i]['location'][min_idx][0],
                                       tracking_seq[i]['location'][min_idx][1]]
                if min_dis < threshold and min_idx >= 0 and points_this not in points_sampled:
                    points_sampled.append(points_this)
                    # print("add ", points_sampled)
            L = least_square_x_y(points_sampled)
            curr_nodes_num = len(points_sampled)
            if curr_nodes_num - last_nodes_num == 0:
                break


        # check if this trajectory is good.
        # evaluation method: traj_length > p_num && break_time < 2 && connecting_node_num > 0.8 * traj_length
        traj_length = 0
        break_time = 0
        connecting_node_num = 0
        traj_choose_flag = False
        traj_start_flag = True
        traj_start_frame = -1
        traj_end_frame = -1
        last_connect_flag = False
        idx_list = []
        dis_list = []
        for i in range(len(tracking_seq)):
            if tracking_seq[i]['metadata']['image_idx'] > start_idx + args.window_size:
                break
            idx_list.append(-1)
            min_idx, min_dis = evaluate_distance_x_y(tracking_seq[i], L, threshold=100, frame=args.frame)
            dis_list.append(min_dis)
            if min_idx >= 0:
                if args.frame == 'global':
                    points_this = [tracking_seq[i]['metadata']['image_idx'],
                                   tracking_seq[i]['global_location'][min_idx][0],
                                   tracking_seq[i]['global_location'][min_idx][1]]
                else:
                    points_this = [tracking_seq[i]['metadata']['image_idx'],
                                   tracking_seq[i]['location'][min_idx][0],
                                   tracking_seq[i]['location'][min_idx][1]]
            if min_dis < threshold and min_idx >= 0 and points_this in points:
                idx_list[i] = min_idx
                dis_list[i] = min_dis
                traj_end_frame = i
                connecting_node_num += 1
                if traj_start_flag:
                    traj_start_flag = False
                    traj_start_frame = i
                last_connect_flag = True
            else:
                if last_connect_flag:
                    break_time += 1
                last_connect_flag = False
        traj_length = traj_end_frame - traj_start_frame + 1
        if traj_length > p_num and break_time < 2 and connecting_node_num > 0.8 * traj_length:
            traj_choose_flag = True
        #     print("L is good: ", L)
        # else:
        #     print("L is not good")
        #     print("traj_length: ", traj_length)
        #     print("break_time: ", break_time)
        #     print("connecting_node_num: ", connecting_node_num)

        if traj_choose_flag:
            l = l + 1
            l_i = l
            lines.append(L)

            # now try to connected old trajectory.
            # if there are nodes which has track id > 0, then use old track id.
            for i in range(len(tracking_seq)):
                if idx_list[i] < 0:
                    continue
                if args.frame == 'global':
                    points_this = [tracking_seq[i]['metadata']['image_idx'],
                                   tracking_seq[i]['global_location'][idx_list[i]][0],
                                   tracking_seq[i]['global_location'][idx_list[i]][1]]
                else:
                    points_this = [tracking_seq[i]['metadata']['image_idx'],
                                   tracking_seq[i]['location'][idx_list[i]][0],
                                   tracking_seq[i]['location'][idx_list[i]][1]]
                if idx_list[i] >= 0 and points_this in points:
                    if traj_seq[i + start_idx]['track_id'][idx_list[i]] >=0:
                        l_i = traj_seq[i + start_idx]['track_id'][idx_list[i]]
                        l = l - 1
                        break

            print("in sequence %d, frame_idx from %d, this is trajectory #%d / %d" % (tracking_seq[0]['metadata']['image_idx'], start_idx, l_i, l))
            print("idx_list = ", idx_list)
            print("dis_list = ", dis_list)
            print("traj_length: ", traj_length)
            print("break_time: ", break_time)
            print("connecting_node_num: ", connecting_node_num)
            print(" ")
            for i in range(len(tracking_seq)):
                if idx_list[i] < 0:
                    continue
                if args.frame == 'global':
                    points_this = [tracking_seq[i]['metadata']['image_idx'],
                                   tracking_seq[i]['global_location'][idx_list[i]][0],
                                   tracking_seq[i]['global_location'][idx_list[i]][1]]
                else:
                    points_this = [tracking_seq[i]['metadata']['image_idx'],
                                   tracking_seq[i]['location'][idx_list[i]][0],
                                   tracking_seq[i]['location'][idx_list[i]][1]]
                if idx_list[i] >= 0 and points_this in points:
                    tracking_seq[i]['track_id'][idx_list[i]] = l_i
                    # print("\n\n\ntracking_seq[i]: ", tracking_seq[i])
                    # print("\ntraj_seq[i + start_idx]: ", traj_seq[i + start_idx])
                    # print("\ni: ", i)
                    # print("\nstart_idx: ", start_idx)
                    traj_seq[i + start_idx]['track_id'][idx_list[i]] = l_i
                    # print('traj_seq[',i + start_idx,'][\'track_id\'][',idx_list[i],'] = ',l_i)

                    # print("before remove, points:", points)
                    # print("remove: ", points_this)
                    points.remove(points_this)
                    # print("after remove, points: ", points)
                    # if i + start_idx >= 96:
                    #     IPython.embed()

                    # make sure all points in this time stamp has different track id
                    for tt in range(len(traj_seq[i + start_idx]['track_id'])):
                        if tt != idx_list[i] and traj_seq[i + start_idx]['track_id'][tt] == l_i:
                            traj_seq[i + start_idx]['track_id'][tt] = -1
                        # if tt != idx_list[i] and tracking_seq[i]['track_id'][tt] == l_i:
                        #     tracking_seq[i]['track_id'][tt] = -1

            # pred_trajectories = get_trajectory(tracking_seq, T_cam_velo, pose_seq, type_whitelist, frame=args.frame)
            # draw_3d(origin_points, pred_trajectories, None, lines=L, vis=vis)
    # if start_idx > 90:
    # pred_trajectories = get_trajectory(tracking_seq, T_cam_velo, pose_seq, type_whitelist, frame=args.frame)
    # draw_3d(origin_points, pred_trajectories, None, lines=lines, vis=vis)
    # print("\n\n")
    return l


def save_trajectory_to_KITTI_format(traj, save_path):
    file = open(save_path, 'w')
    for i in range(len(traj)):
        for t in range(len(traj[i]['track_id'])):
            if traj[i]['track_id'][t] > 0:
                file.write(str(int(traj[i]['metadata']['image_idx'])))
                file.write(' ')
                file.write(str(int(traj[i]['track_id'][t])))
                file.write(' ')
                type = 'Car'
                file.write(type)
                file.write(' ')
                file.write(str(round(0, 6)))
                file.write(' ')
                file.write(str(round(0, 6)))
                file.write(' ')
                file.write(str(round(traj[i]['alpha'][t], 6)))
                file.write(' ')
                left = traj[i]['bbox'][t][0]
                top = traj[i]['bbox'][t][1]
                right = traj[i]['bbox'][t][2]
                bottom = traj[i]['bbox'][t][3]
                file.write(str(round(left, 6)))
                file.write(' ')
                file.write(str(round(top, 6)))
                file.write(' ')
                file.write(str(round(right, 6)))
                file.write(' ')
                file.write(str(round(bottom, 6)))
                file.write(' ')
                dimensions = traj[i]['dimensions'][t]
                file.write(str(round(dimensions[2], 6)))
                file.write(' ')
                file.write(str(round(dimensions[1], 6)))
                file.write(' ')
                file.write(str(round(dimensions[0], 6)))
                assert np.all(dimensions != 0)
                file.write(' ')
                location = traj[i]['location'][t]
                file.write(str(round(location[0], 6)))
                file.write(' ')
                file.write(str(round(location[1], 6)))
                file.write(' ')
                file.write(str(round(location[2], 6)))
                file.write(' ')
                file.write(str(round(traj[i]['rotation_y'][t], 6)))
                file.write('\n')

    file.close()
    print("save finished into ", save_path)



trajectory_result = []
for seq in range(args.seq_start, args.seq_end + 1):
    if seq == 1:
        print("Do not compose seq 1, because lack 4 point clouds.")
        continue
    # output_dir = os.path.join(args.output_dir, '%4d' % seq)
    # if not os.path.exists(output_dir):
    #     os.makedirs(output_dir)

    # get calib
    calib_path = os.path.join(args.data_dir, "calib", '%04d.txt' % seq)
    calib_seq = KittiCalib(calib_path).read_calib_file()
    T_cam_velo = calib_seq.Tr_cam_to_velo
    # get pose
    pose_seq = load_pose(args.velodyne_dir, args.pose_dir, seq)

    l = 0
    det_seq = []
    points_seq = []
    for i in range(len(dt_data)):
        if dt_data[i]['metadata']['image_seq'] < seq:
            continue
        if dt_data[i]['metadata']['image_seq'] > seq:
            break
        det_seq.append(dt_data[i])
        if args.frame == 'global':
            centers_global = dt_data[i]['global_location']
        else:
            centers_global = dt_data[i]['location']
        t = dt_data[i]['metadata']['image_idx']
        for c in range(len(centers_global)):
            [x_global, y_global, z_global] = centers_global[c]
            points_seq.append([t, x_global, y_global])
    traj_seq = copy.deepcopy(det_seq)
    for i in range(len(traj_seq)):
        traj_seq[i]['track_id'] = np.ones_like(traj_seq[i]['score']) * -1

    # map_num = math.ceil(len(det_seq) / args.map_duration_frame)
    STEP_MAX_ = 999999
    for step in range(0, STEP_MAX_):
        start_idx = args.step_size * step
        end_idx = start_idx + args.window_size
        # get the detection result in this sequence
        det_window = []
        for i in range(len(dt_data)):
            if dt_data[i]['metadata']['image_seq'] < seq:
                continue
            if dt_data[i]['metadata']['image_seq'] > seq:
                break
            if dt_data[i]['metadata']['image_idx'] < start_idx:
                continue
            if dt_data[i]['metadata']['image_idx'] >= end_idx:
                break
            det_window.append(dt_data[i])

        if len(det_window) < 3: break

        # print("det_window: ", det_window)

        spatio_temporal_t_x_y_map_points = []
        for j in range(len(det_window)):
            if args.frame == 'global':
                centers_global = det_window[j]['global_location']
            else:
                centers_global = det_window[j]['location']
            t = det_window[j]['metadata']['image_idx']
            for c in range(len(centers_global)):
                [x_global, y_global, z_global] = centers_global[c]
                spatio_temporal_t_x_y_map_points.append([t, x_global, y_global])

        # pred_trajectories = get_trajectory(tracking_seq, T_cam_velo, pose_seq, type_whitelist, frame=args.frame)
        # label_trajectories = get_trajectory(tracking_label_seq, T_cam_velo, pose_seq, type_whitelist, frame=args.frame)
        # draw_3d(spatio_temporal_t_x_y_map_points, None, None, vis=True)
        # print('spatio_temporal_t_x_y_map_points: ', spatio_temporal_t_x_y_map_points)
        l = ransac_find_trajectory(traj_seq, start_idx, l, det_window, spatio_temporal_t_x_y_map_points, args.p_num, args.iterations, args.RANSAC_threshold, vis=True)

        # find_trajectory(det_seq, spatio_temporal_t_x_y_map_points, vis=True)

        print(start_idx, end_idx)
        if end_idx > det_seq[-1]['metadata']['image_idx']:
            break
    # pred_trajectories = get_trajectory(traj_seq, T_cam_velo, pose_seq, type_whitelist, frame=args.frame)
    # draw_3d(points_seq, pred_trajectories, None, None, vis=True)

    if args.tracking_label_save_flag:
        # save pred trajectory KITTI format
        save_dir = os.path.join(args.tracking_label_save_dir, "pred")
        if not os.path.exists(save_dir): os.mkdir(save_dir)
        save_path = os.path.join(save_dir, "%04d.txt" % seq)
        save_trajectory_to_KITTI_format(traj_seq, save_path)