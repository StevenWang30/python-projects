import numpy as np
from tracking_utils.dataset import *
# from tracking_utils.tf_image_vis import *
import IPython
import matplotlib.pyplot as plt
from tracking_utils.utils import *
import random
from tracking_utils.sort import sort
import open3d as o3d
from tracking_utils.kitti_tracking_utils import *
from tracking_utils.box_util import box3d_iou
# import lib.tracking_utils.dataset as dataset

class ObjectData:
    def __init__(self):
        object_dict = dict()
        object_dict['frame_ID'] = np.zeros(0)
        object_dict['object_ID'] = np.zeros(0)
        object_dict['type'] = []
        # object_dict['truncation'] = np.zeros(0)
        # object_dict['occlusion'] = np.zeros(0)
        object_dict['alpha'] = np.zeros(0)
        object_dict['left'] = np.zeros(0)
        object_dict['top'] = np.zeros(0)
        object_dict['right'] = np.zeros(0)
        object_dict['bottom'] = np.zeros(0)

        object_dict['lwh'] = np.zeros((0, 3))
        object_dict['xyz'] = np.zeros((0, 3))
        object_dict['yaw'] = np.zeros(0)
        object_dict['start_frame'] = np.zeros(0)
        object_dict['fresh_frame'] = np.zeros(0)
        object_dict['fresh_times'] = np.zeros(0)
        object_dict['score'] = np.zeros(0)
        object_dict['last_T_cam_velo'] = 0
        object_dict['last_object_num'] = 0
        object_dict['max_object_num'] = 0
        object_dict['frame2_bboxes_clean_times'] = 0

        self.data = object_dict

    def __len__(self):
        if not self.data['frame_ID'].shape[0] == self.data['object_ID'].shape[0] == self.data['lwh'].shape[0] == self.data['xyz'].shape[0] ==\
                self.data['yaw'].shape[0] == self.data['start_frame'].shape[0] == self.data['fresh_frame'].shape[0] == self.data['fresh_times'].shape[0] ==\
                self.data['score'].shape[0] == len(self.data['type']):
            IPython.embed()
        return self.data['score'].shape[0]

    def decay(self, score):
        return 0.85 * score

    def init_first_object_data(self, frame, center, size, heading_angle, sem_class, score, T_cam_velo):
        bboxes_num = center.shape[0]
        self.data['frame_ID'] = frame + np.zeros(bboxes_num)
        # self.data['pc_center'] = [pc_center] + np.zeros((bboxes_num, 3))
        self.data['start_frame'] = frame + np.zeros(bboxes_num)
        self.data['fresh_frame'] = frame + np.zeros(bboxes_num)
        self.data['fresh_times'] = np.zeros(bboxes_num)
        self.data['object_ID'] = np.array(range(bboxes_num))
        for i in range(bboxes_num):
            self.data['type'].append(class2type[sem_class[i]])
        # self.data['truncation'] = np.zeros(bboxes_num)
        # self.data['occlusion'] = np.zeros(bboxes_num)

        self.data['lwh'] = size
        self.data['xyz'] = center
        self.data['yaw'] = heading_angle
        self.data['score'] = score
        self.data['last_object_num'] = bboxes_num
        self.data['max_object_num'] = bboxes_num
        self.data['last_T_cam_velo'] = T_cam_velo

    def update_object_data_compare(self, pointcloud_f1, pointcloud_f2, pc1_chosen, tracking_pred, mask_f1,
                                center_f1, size_f1, eading_angle_f1, sem_class_f1, score_f1,
                                frame_f2, center_f2, size_f2, heading_angle_f2, sem_class_f2, score_f2, T_cam_velo_f2,
                           associate_method='SORT', threshold=4.0):
        associate_method_list = ['SORT', 'DEEP_SORT', 'OURS+SORT', 'OURS+DEEP_SORT', 'GT+SORT', 'GT+DEEP_SORT']
        DEEP_SORT_ALPHA = 1
        DEEP_SORT_BETA = 1
        SCORE_SCLAE = 1
        if associate_method not in associate_method:
            print("The association method must in the prepared list ", associate_method_list)
            assert True

        bb_num_f1 = self.data['last_object_num']
        f1_start_idx = len(self) - self.data['last_object_num']
        size_f1 = self.data['lwh'][f1_start_idx:].copy()
        center_f1 = self.data['xyz'][f1_start_idx:].copy()
        center_f1_origin = self.data['xyz'][f1_start_idx:].copy()
        heading_f1 = self.data['yaw'][f1_start_idx:].copy()
        score_f1 = self.data['score'][f1_start_idx:].copy()
        start_frame_f1 = self.data['start_frame'][f1_start_idx:].copy()
        T_cam_velo_f1 = self.data['last_T_cam_velo'].copy()
        object_ID_f1 = self.data['object_ID'][f1_start_idx:].copy()
        fresh_frame_f1 = self.data['fresh_frame'][f1_start_idx:].copy()
        fresh_times_f1 = self.data['fresh_times'][f1_start_idx:].copy()

        if associate_method in ['OURS+SORT', 'OURS+DEEP_SORT', 'GT+SORT', 'GT+DEEP_SORT']:
            if associate_method in ['OURS+SORT', 'OURS+DEEP_SORT']:
                pointcloud = pc1_chosen
            elif associate_method in ['GT+SORT', 'GT+DEEP_SORT']:
                pointcloud = pointcloud_f1
            if bb_num_f1 > 0:
                for bb in range(bb_num_f1):
                    idx = find_points_idx_in_bbox(pointcloud, center_f1[bb], size_f1[bb], heading_f1[bb],
                                                  T_cam_velo_f1, threshold=1.0)

                    if sum(idx.astype(np.float32)) < 20:
                        # print('points in bbox ', bb, ' is ', sum(idx.astype(np.float32)), ', continue.')
                        continue

                    tracking_bb_this = tracking_pred[idx]
                    # score_f1_this = score_nms_f1[idx]
                    if tracking_bb_this.shape[0] > 0:
                        # max_order = tracking_bb_this.argsort()[::-1]
                        # tracking_bb_this = tracking_bb_this[max_order[:10]]
                        # IPython.embed()
                        tracking_mean = np.mean(tracking_bb_this, 0)
                        # print('in bbox ', object_ID_f1[bb], ', add tracking_mean is ', tracking_mean)
                    else:
                        tracking_mean = np.zeros(7)
                        # print('add tracking 0000')
                    center_f1[bb] += tracking_mean[:3]

        # tracked_bboxes_points_f1 = np.zeros((0,3))
        # for i in range(bb_num_f1):
        #     tracked_bboxes_points_i = get_bboxes_points(size_f1[i], heading_f1[i], center_f1[i], T_cam_velo_f1)
        #     tracked_bboxes_points_i = draw_birdeye_bboxes(tracked_bboxes_points_i)
        #     tracked_bboxes_points_f1 = np.append(tracked_bboxes_points_f1, tracked_bboxes_points_i, axis=0)
        #
        # bboxes_points_f1 = np.zeros((0, 3))
        # for i in range(center_f1_origin.shape[0]):
        #     bboxes_points_i = get_bboxes_points(size_f1[i], heading_f1[i], center_f1_origin[i], T_cam_velo_f1)
        #     bboxes_points_i = draw_birdeye_bboxes(bboxes_points_i)
        #     bboxes_points_f1 = np.append(bboxes_points_f1, bboxes_points_i, axis=0)
        #
        # bboxes_points_f2 = np.zeros((0, 3))
        # for i in range(center_f2.shape[0]):
        #     bboxes_points_i = get_bboxes_points(size_f2[i], heading_angle_f2[i], center_f2[i], T_cam_velo_f1)
        #     bboxes_points_i = draw_birdeye_bboxes(bboxes_points_i)
        #     bboxes_points_f2 = np.append(bboxes_points_f2, bboxes_points_i, axis=0)
        # fig = plt.figure()
        # colors = {'g': '#008000', 'r': '#FF0000', 'b': '#0000FF', 'm': '#FF00FF'}
        # plt.scatter(bboxes_points_f1[:, 0], bboxes_points_f1[:, 1], c=colors['r'], s=0.2, alpha=0.6)
        # plt.scatter(bboxes_points_f2[:, 0], bboxes_points_f2[:, 1], c=colors['b'], s=0.2, alpha=0.5)
        # plt.scatter(tracked_bboxes_points_f1[:, 0], tracked_bboxes_points_f1[:, 1], c=colors['m'], s=0.2, alpha=0.5)
        # plt.savefig('/data/KITTI_object_tracking/experiments/pred/pcd_vis/' + associate_method+str(frame_f2)+'.png', dpi=500)
        # plt.close('all')  # 关闭图 0


        # calculate similarity matrix
        if center_f1.shape[0] == 0 or center_f2.shape[0] == 0:
            matches = np.zeros(0)
            unmatched_frame_1 = np.array(range(center_f1.shape[0]))
            unmatched_frame_2 = np.array(range(center_f2.shape[0]))
        else:
            if associate_method in ['OURS+SORT', 'SORT', 'GT+SORT']:
                dist_mat = np.sum(np.sqrt((np.expand_dims(center_f1, 1) - np.expand_dims(center_f2, 0))**2), axis=-1)
                # bboxes_assignment = np.argmin(dist_mat, axis=-1)  # B * PR
                # min_dist = np.reduce_min(dist_mat, axis=-1)  # B * PR



                # IPython.embed()

                score_dif_mat = (np.expand_dims(score_f1, 1) - np.expand_dims(score_f2, 0)) * SCORE_SCLAE

                cost_matrix = dist_mat + score_dif_mat












            elif associate_method in ['OURS+DEEP_SORT', 'DEEP_SORT', 'GT+DEEP_SORT']:
                dist_mat = np.sum(np.sqrt((np.expand_dims(center_f1, 1) - np.expand_dims(center_f2, 0)) ** 2), axis=-1)
                feat_f1 = np.append(size_f1, np.expand_dims(heading_f1, -1), axis=-1)
                feat_f1 = np.append(feat_f1, np.expand_dims(score_f1, -1), axis=-1)
                feat_f2 = np.append(size_f2, np.expand_dims(heading_angle_f2, -1), axis=-1)
                feat_f2 = np.append(feat_f2, np.expand_dims(score_f2, -1), axis=-1)
                feat_mat = np.sum(np.sqrt((np.expand_dims(feat_f1, 1) - np.expand_dims(feat_f2, 0)) ** 2), axis=-1)

                cost_matrix = DEEP_SORT_ALPHA * dist_mat + DEEP_SORT_BETA * feat_mat

                score_dif_mat = (np.expand_dims(score_f1, 1) - np.expand_dims(score_f2, 0)) * SCORE_SCLAE

                cost_matrix = cost_matrix + score_dif_mat
            else:
                assert True
            # SORT
            matches, unmatched_frame_1, unmatched_frame_2 = sort(cost_matrix, threshold=8.0)

        # UPDATE
        current_object_num = 0
        for i in range(matches.shape[0]):
            frame1_idx = matches[i, 0]
            frame2_idx = matches[i, 1]
            score_decay = self.decay(score_f1[frame1_idx])
            score_merge = max(score_decay, score_f2[frame2_idx])
            self.data['frame_ID'] = np.append(self.data['frame_ID'], frame_f2)
            # same as the last object ID
            self.data['object_ID'] = np.append(self.data['object_ID'], object_ID_f1[frame1_idx])
            self.data['lwh'] = np.append(self.data['lwh'], [size_f2[frame2_idx]], axis=0)
            self.data['xyz'] = np.append(self.data['xyz'], [center_f2[frame2_idx]], axis=0)
            self.data['yaw'] = np.append(self.data['yaw'], heading_angle_f2[frame2_idx])
            # if score_f2[frame2_idx] > 0.85:
            self.data['score'] = np.append(self.data['score'], min(score_merge * 10 / 8, 1.0 - 1e-7))
            self.data['start_frame'] = np.append(self.data['start_frame'], start_frame_f1[frame1_idx])
            self.data['fresh_frame'] = np.append(self.data['fresh_frame'], frame_f2)
            self.data['fresh_times'] = np.append(self.data['fresh_times'], fresh_times_f1[frame1_idx] + 1)
            self.data['type'].append(self.data['type'][f1_start_idx + frame1_idx])
            current_object_num += 1

        for i in range(unmatched_frame_2.shape[0]):
            frame2_idx = unmatched_frame_2[i]
            # add new f2 bb.
            self.data['frame_ID'] = np.append(self.data['frame_ID'], frame_f2)
            # self.data['pc_center'] = np.append(self.data['pc_center'], [pc_center_f2], axis=0)
            self.data['start_frame'] = np.append(self.data['start_frame'], frame_f2)
            self.data['fresh_frame'] = np.append(self.data['fresh_frame'], frame_f2)
            self.data['fresh_times'] = np.append(self.data['fresh_times'], 0)
            self.data['max_object_num'] += 1
            self.data['object_ID'] = np.append(self.data['object_ID'], self.data['max_object_num'] - 1)

            self.data['lwh'] = np.append(self.data['lwh'], [size_f2[frame2_idx]], axis=0)
            self.data['xyz'] = np.append(self.data['xyz'], [center_f2[frame2_idx]], axis=0)
            self.data['yaw'] = np.append(self.data['yaw'], heading_angle_f2[frame2_idx])
            self.data['score'] = np.append(self.data['score'], score_f2[frame2_idx])
            self.data['type'].append(class2type[sem_class_f2[frame2_idx]])
            current_object_num += 1

        # if associate_method in ['OURS+SORT', 'OURS+DEEP_SORT', 'GT+SORT', 'GT+DEEP_SORT']:
        #     # IPython.embed()
        #     # print(unmatched_frame_1)
        #     # if frame_f2 > 425:
        #     #     IPython.embed()
        for i in range(unmatched_frame_1.shape[0]):
            if center_f2.shape[0] > 0:
                if min(cost_matrix[unmatched_frame_1[i]]) < 5:
                    idx_f1 = unmatched_frame_1[i]
                    nearest_idx = np.argmin(cost_matrix[unmatched_frame_1[i]])
                    bbox_points_f1_this = get_bboxes_points(size_f1[idx_f1], heading_f1[idx_f1], center_f1[idx_f1], T_cam_velo_f1)
                    bbox_points_f2_this = get_bboxes_points(size_f2[nearest_idx], heading_angle_f2[nearest_idx],
                                                            center_f2[nearest_idx], T_cam_velo_f1)
                    bbox_points_a = transform_points(bbox_points_f1_this, np.linalg.inv(T_cam_velo_f1))
                    bbox_points_b = transform_points(bbox_points_f2_this, np.linalg.inv(T_cam_velo_f1))
                    # print(bbox_points_a, bbox_points_b)
                    ''' compare the iou2d, not iou3d'''
                    iou_3d, iou_2d = box3d_iou(bbox_points_a, bbox_points_b)
                    if iou_2d > 0.01:
                        continue
            frame1_idx = unmatched_frame_1[i]
            score_decay = self.decay(score_f1[frame1_idx])
            if score_decay < 0.6:
                # drop this object now.
                continue
            self.data['frame_ID'] = np.append(self.data['frame_ID'], frame_f2)
            # same as the last object ID
            self.data['object_ID'] = np.append(self.data['object_ID'], object_ID_f1[frame1_idx])
            self.data['lwh'] = np.append(self.data['lwh'], [size_f1[frame1_idx]], axis=0)
            self.data['xyz'] = np.append(self.data['xyz'], [center_f1[frame1_idx]], axis=0)
            self.data['yaw'] = np.append(self.data['yaw'], heading_f1[frame1_idx])
            self.data['score'] = np.append(self.data['score'], score_decay)
            self.data['start_frame'] = np.append(self.data['start_frame'], start_frame_f1[frame1_idx])
            self.data['fresh_frame'] = np.append(self.data['fresh_frame'], fresh_frame_f1[frame1_idx])
            self.data['fresh_times'] = np.append(self.data['fresh_times'], fresh_times_f1[frame1_idx])
            self.data['type'].append(self.data['type'][f1_start_idx + frame1_idx])
            current_object_num += 1

        self.data['last_object_num'] = current_object_num

        # print("Current frame has ", self.data['last_object_num'], " objects.")

    def update_object_data(self, pointcloud_f1, pointcloud_f2, pc1_chosen, tracking_pred, mask_f1,
                                center_f1_detection, size_f1_detection, heading_f1_detection, sem_class_f1_detection, score_f1_detection,
                                frame_f2, center_f2, size_f2, heading_angle_f2, sem_class_f2, score_f2, T_cam_velo_f2,
                           use_tracking=True):
        # using frame1 detection to refine the self.data
        bb_num_f1 = self.data['last_object_num']
        f1_start_idx = len(self) - self.data['last_object_num']
        size_f1 = self.data['lwh'][f1_start_idx:]
        center_f1 = self.data['xyz'][f1_start_idx:]
        heading_f1 = self.data['yaw'][f1_start_idx:]
        T_cam_velo_f1 = self.data['last_T_cam_velo']
        score_f1 = self.data['score'][f1_start_idx:]

        bb_num_f1_detection = center_f1_detection.shape[0]
        # TODO: consider the self.data's fresh times information, if the bboxes in the data have updated, then do not erase.
        chosen_list = []
        for i in range(bb_num_f1_detection):
            min_dis = 999
            min_dis_index = -1
            for j in range(bb_num_f1):
                dis = math.sqrt(np.sum((center_f1_detection[i] - center_f1[j]) * (center_f1_detection[i] - center_f1[j])))
                if dis < min_dis:
                    min_dis_index = j
                    min_dis = dis

            iou_2d = 0
            if min_dis_index != -1:
                chosen_list.append(min_dis_index)
                if not (size_f1_detection[i][0] < 2 or np.any(size_f1_detection[i] < 0.01) or size_f1[min_dis_index][0] < 2 or np.any(size_f1[min_dis_index] < 0.01)):
                    bbox_points_f1_this = get_bboxes_points(size_f1_detection[i], heading_f1_detection[i], center_f1_detection[i], T_cam_velo_f1)
                    bbox_points_f2_this = get_bboxes_points(size_f1[min_dis_index], heading_f1[min_dis_index], center_f1[min_dis_index], T_cam_velo_f1)
                    bbox_points_a = transform_points(bbox_points_f1_this, np.linalg.inv(T_cam_velo_f1))
                    bbox_points_b = transform_points(bbox_points_f2_this, np.linalg.inv(T_cam_velo_f1))
                    # print(bbox_points_a, bbox_points_b)
                    ''' compare the iou2d, not iou3d'''
                    iou_3d, iou_2d = box3d_iou(bbox_points_a, bbox_points_b)
                    if iou_2d > 0.01:
                        # frame1 and data all have
                        center_f1[min_dis_index] = (center_f1[min_dis_index] + center_f1_detection[i]) / 2

            if min_dis_index == -1 or iou_2d <= 0.01:
                # frame1 have but data doesn't have
                if score_f1_detection[i] > 0.8:
                    self.data['frame_ID'] = np.append(self.data['frame_ID'], frame_f2-1)
                    # self.data['pc_center'] = np.append(self.data['pc_center'], [pc_center_f2], axis=0)
                    self.data['last_object_num'] += 1
                    self.data['start_frame'] = np.append(self.data['start_frame'], frame_f2-1)
                    self.data['fresh_frame'] = np.append(self.data['fresh_frame'], frame_f2-1)
                    self.data['fresh_times'] = np.append(self.data['fresh_times'], 0)
                    self.data['max_object_num'] += 1
                    self.data['object_ID'] = np.append(self.data['object_ID'], self.data['max_object_num'] - 1)
                    self.data['lwh'] = np.append(self.data['lwh'], [size_f1_detection[i]], axis=0)
                    self.data['xyz'] = np.append(self.data['xyz'], [center_f1_detection[i]], axis=0)
                    self.data['yaw'] = np.append(self.data['yaw'], heading_f1_detection[i])
                    self.data['score'] = np.append(self.data['score'], score_f1_detection[i])
                    self.data['type'].append(class2type[sem_class_f1_detection[i]])
        disp = 0
        for i in range(bb_num_f1):
            if not i in chosen_list:
                if score_f1[i] < 0.7:
                    self.data['frame_ID'] = np.delete(self.data['frame_ID'], f1_start_idx+i-disp, axis=0)
                    self.data['start_frame'] = np.delete(self.data['start_frame'], f1_start_idx+i-disp, axis=0)
                    self.data['fresh_frame'] = np.delete(self.data['fresh_frame'], f1_start_idx+i-disp, axis=0)
                    self.data['fresh_times'] = np.delete(self.data['fresh_times'], f1_start_idx+i-disp, axis=0)
                    self.data['object_ID'] = np.delete(self.data['object_ID'], f1_start_idx+i-disp, axis=0)

                    self.data['lwh'] = np.delete(self.data['lwh'], f1_start_idx+i-disp, axis=0)
                    self.data['xyz'] = np.delete(self.data['xyz'], f1_start_idx+i-disp, axis=0)
                    self.data['yaw'] = np.delete(self.data['yaw'], f1_start_idx+i-disp, axis=0)
                    self.data['score'] = np.delete(self.data['score'], f1_start_idx+i-disp, axis=0)
                    # IPython.embed()
                    # print(f1_start_idx, bb_num_f1, f1_start_idx+i-disp, len(self.data['type']), chosen_list)
                    del self.data['type'][f1_start_idx+i-disp]
                    self.data['last_object_num'] -= 1
                    disp += 1

        bb_num_f1 = self.data['last_object_num']
        f1_start_idx = len(self) - self.data['last_object_num']
        size_f1 = self.data['lwh'][f1_start_idx:].copy()
        center_f1 = self.data['xyz'][f1_start_idx:].copy()
        heading_f1 = self.data['yaw'][f1_start_idx:].copy()
        score_f1 = self.data['score'][f1_start_idx:].copy()
        start_frame_f1 = self.data['start_frame'][f1_start_idx:].copy()
        T_cam_velo_f1 = self.data['last_T_cam_velo'].copy()
        object_ID_f1 = self.data['object_ID'][f1_start_idx:].copy()
        fresh_frame_f1 = self.data['fresh_frame'][f1_start_idx:].copy()
        fresh_times_f1 = self.data['fresh_times'][f1_start_idx:].copy()

        positive_idx = np.where(mask_f1[..., 0] > mask_f1[..., 1])
        pointcloud_f1 = pointcloud_f1[positive_idx]
        # tracking_pred = tracking_pred[positive_idx]
        score_nms_f1 = mask_f1[positive_idx][:, 0]

        # survive_list = []
        # for bb_idx2 in range(center_f2.shape[0]):
        #     idx = find_points_idx_in_bbox(pointcloud_f2, center_f2[bb_idx2], size_f2[bb_idx2],
        #                                   heading_angle_f2[bb_idx2], T_cam_velo_f1, threshold=3.0)
        #     if not (sum(idx.astype(np.float32)) < 5 and score_f2[bb_idx2] < 0.7):
        #         survive_list.append(bb_idx2)
        #         # print('survive bbox ', bb_idx2, ' with pr', pr, '/vanish_rate', vanish_rate)
        #     # else:
        # print('In frame ', frame_f2, ', clean ', center_f2.shape[0] - len(survive_list), ' bboxes.')
        # self.data['frame2_bboxes_clean_times'] = self.data['frame2_bboxes_clean_times'] + center_f2.shape[0] - len(
        #     survive_list)
        # center_f2 = center_f2[survive_list]
        # size_f2 = size_f2[survive_list]
        # heading_angle_f2 = heading_angle_f2[survive_list]
        # sem_class_f2 = sem_class_f2[survive_list]
        # score_f2 = score_f2[survive_list]

        if use_tracking:
            if bb_num_f1 > 0:

                for bb in range(bb_num_f1):
                    idx = find_points_idx_in_bbox(pc1_chosen, center_f1[bb], size_f1[bb], heading_f1[bb],
                                                  T_cam_velo_f1)

                    if sum(idx.astype(np.float32)) < 10:
                        # print('points in bbox ', bb, ' is ', sum(idx.astype(np.float32)), ', continue.')
                        continue

                    tracking_bb_this = tracking_pred[idx]
                    # score_f1_this = score_nms_f1[idx]
                    if tracking_bb_this.shape[0] > 0:
                        # order_f1 = score_f1_this.argsort()[::-1]
                        # tracking_bb_this = tracking_bb_this[order_f1[:order_f1.shape[0] // 3]]
                        tracking_mean = np.mean(tracking_bb_this, 0)
                        # print('in bbox ', object_ID_f1[bb], ', add tracking_mean is ', tracking_mean)
                    else:
                        tracking_mean = np.zeros(7)
                        # print('add tracking 0000')
                    # print(center_f1[bb])
                    center_f1[bb] += tracking_mean[:3]
                    # print(center_f1[bb])
                    # print('center ', bb, ' add ', tracking_mean[:3])
                    # size_f1[bb] += tracking_mean[3:6]
                    # heading_f1[bb] += tracking_mean[6]

        # print("size_f1 nana? second", np.isnan(np.sum(size_f1)))
        # frame_f2 = self.data['frame_ID'][-1] + 1
        self.data['last_T_cam_velo'] = T_cam_velo_f2
        current_object_num = 0
        bb_num_f1 = center_f1.shape[0]
        bb_num_f2 = center_f2.shape[0]
        chosen_idx = []
        chosen_ID = []
        # print("last frame (frame ", str(frame_f2 - 1), ") has ", bb_num_f1, " frame.")
        for i in range(bb_num_f1):
            score_decay = self.decay(score_f1[i])
            bbox_points_f1_this = get_bboxes_points(size_f1[i], heading_f1[i], center_f1[i], T_cam_velo_f1)
            max_iou = -1
            max_idx = -1
            min_dis = 999
            min_dis_ind = -1
            for j in range(bb_num_f2):
                if j in chosen_idx:
                    continue
                distance = math.sqrt(np.sum((center_f1[i] - center_f2[j]) * (center_f1[i] - center_f2[j])))
                if distance < min_dis:
                    min_dis = distance
                    min_dis_ind = j
            if min_dis_ind != -1:
                bbox_points_f2_this = get_bboxes_points(size_f2[min_dis_ind], heading_angle_f2[min_dis_ind], center_f2[min_dis_ind],
                                                        T_cam_velo_f2)
                bbox_points_a = transform_points(bbox_points_f1_this, np.linalg.inv(T_cam_velo_f1))
                bbox_points_b = transform_points(bbox_points_f2_this, np.linalg.inv(T_cam_velo_f2))
                ''' compare the iou2d, not iou3d'''
                iou_3d, iou_2d = box3d_iou(bbox_points_a, bbox_points_b)
                max_iou = iou_2d
                max_idx = min_dis_ind

            # IPython.embed()
            if max_iou > 0.01:
                if max(score_f2[max_idx], score_decay) < 0.5:
                    continue
                chosen_idx.append(max_idx)

                # judge and update score
                if score_f2[max_idx] > 0.85 or score_f2[max_idx] - score_decay > 0:
                    # add origin f2 bb
                    # print("size_f2maxidx", size_f2[max_idx])
                    # update frame id and add last num
                    self.data['frame_ID'] = np.append(self.data['frame_ID'], frame_f2)
                    # self.data['pc_center'] = np.append(self.data['pc_center'], [pc_center_f2], axis=0)
                    # self.data['last_object_num'] += 1
                    current_object_num += 1
                    # same as the last object ID
                    self.data['object_ID'] = np.append(self.data['object_ID'], object_ID_f1[i])
                    chosen_ID.append(object_ID_f1[i])
                    self.data['lwh'] = np.append(self.data['lwh'], [size_f2[max_idx]], axis=0)
                    self.data['xyz'] = np.append(self.data['xyz'], [center_f2[max_idx]], axis=0)
                    self.data['yaw'] = np.append(self.data['yaw'], heading_angle_f2[max_idx])
                    # if score_f2[max_idx] > 0.85:
                    self.data['score'] = np.append(self.data['score'], min(score_f2[max_idx] * 10 / 8, 1.0 - 1e-7))
                    # else:
                    #     self.data['score'] = np.append(self.data['score'], score_f2[max_idx])
                    self.data['start_frame'] = np.append(self.data['start_frame'], start_frame_f1[i])
                    self.data['fresh_frame'] = np.append(self.data['fresh_frame'], frame_f2)
                    self.data['fresh_times'] = np.append(self.data['fresh_times'], fresh_times_f1[i] + 1)
                    self.data['type'].append(self.data['type'][f1_start_idx + i])
                elif score_decay - score_f2[max_idx] > 0.3:
                    # add origin f1 bb
                    # print("addf1bb size_f1i", size_f1[i])
                    # update frame id and add last num
                    # print('Frame ', str(frame_f2), ', object ', str(object_ID_f1[i]),
                    #       ' using last frame and tracking(TOTAL)!!!')
                    self.data['frame_ID'] = np.append(self.data['frame_ID'], frame_f2)
                    # self.data['pc_center'] = np.append(self.data['pc_center'], [pc_center_f2], axis=0)
                    # self.data['last_object_num'] += 1
                    current_object_num += 1
                    # same as the last object ID
                    self.data['object_ID'] = np.append(self.data['object_ID'], object_ID_f1[i])
                    chosen_ID.append(object_ID_f1[i])
                    self.data['lwh'] = np.append(self.data['lwh'], [size_f1[i]], axis=0)
                    self.data['xyz'] = np.append(self.data['xyz'], [center_f1[i]], axis=0)
                    self.data['yaw'] = np.append(self.data['yaw'], heading_f1[i])
                    self.data['score'] = np.append(self.data['score'], min(score_decay * 10 / 8, 1.0 - 1e-7))
                    self.data['start_frame'] = np.append(self.data['start_frame'], start_frame_f1[i])
                    self.data['fresh_frame'] = np.append(self.data['fresh_frame'], frame_f2)
                    self.data['fresh_times'] = np.append(self.data['fresh_times'], fresh_times_f1[i] + 1)
                    self.data['type'].append(self.data['type'][f1_start_idx + i])
                else:
                    # add merged f1 and f2 bb
                    # print('Frame ', str(frame_f2), ', object ', str(object_ID_f1[i]),
                    #       ' using last frame and tracking(PART)!!!')
                    score = (score_decay + score_f2[max_idx] * score_f2[max_idx]) / (1 + score_f2[max_idx])
                    # update frame id and add last num
                    self.data['frame_ID'] = np.append(self.data['frame_ID'], frame_f2)
                    # self.data['pc_center'] = np.append(self.data['pc_center'], [pc_center_f2], axis=0)
                    # self.data['last_object_num'] += 1
                    current_object_num += 1
                    # same as the last object ID
                    self.data['object_ID'] = np.append(self.data['object_ID'], object_ID_f1[i])
                    chosen_ID.append(object_ID_f1[i])
                    # print("(size_f1[i] + size_f2[max_idx]) / 2", (size_f1[i] + size_f2[max_idx]) / 2)
                    self.data['lwh'] = np.append(self.data['lwh'], [
                        (size_f1[i] + size_f2[max_idx] * score_f2[max_idx]) / (1 + score_f2[max_idx])], axis=0)
                    # print('in part',(center_f1[i] + center_f2[max_idx] * score_f2[max_idx]) / (1 + score_f2[max_idx]))
                    self.data['xyz'] = np.append(self.data['xyz'], [
                        (center_f1[i] + center_f2[max_idx] * score_f2[max_idx]) / (1 + score_f2[max_idx])], axis=0)
                    self.data['yaw'] = np.append(self.data['yaw'],
                                                 (heading_f1[i] + heading_angle_f2[max_idx] * score_f2[max_idx]) / (
                                                             1 + score_f2[max_idx]))
                    self.data['score'] = np.append(self.data['score'], min(score * 10 / 8, 1.0 - 1e-7))
                    self.data['start_frame'] = np.append(self.data['start_frame'], start_frame_f1[i])
                    self.data['fresh_frame'] = np.append(self.data['fresh_frame'], frame_f2)
                    self.data['fresh_times'] = np.append(self.data['fresh_times'], fresh_times_f1[i] + 1)
                    self.data['type'].append(self.data['type'][f1_start_idx + i])
            else:
                if score_decay < 0.6:
                    # drop this object now.
                    continue
                # check if have iou with any current bboxes.

                # add origin f1 bb
                # print("size_f1i", size_f1[i])
                # update frame id and add last num
                self.data['frame_ID'] = np.append(self.data['frame_ID'], frame_f2)
                # self.data['pc_center'] = np.append(self.data['pc_center'], [pc_center_f2], axis=0)
                # self.data['last_object_num'] += 1
                current_object_num += 1
                # same as the last object ID
                self.data['object_ID'] = np.append(self.data['object_ID'], object_ID_f1[i])
                chosen_ID.append(object_ID_f1[i])
                self.data['lwh'] = np.append(self.data['lwh'], [size_f1[i]], axis=0)
                self.data['xyz'] = np.append(self.data['xyz'], [center_f1[i]], axis=0)
                self.data['yaw'] = np.append(self.data['yaw'], heading_f1[i])
                self.data['score'] = np.append(self.data['score'], score_decay)
                self.data['start_frame'] = np.append(self.data['start_frame'], start_frame_f1[i])
                self.data['fresh_frame'] = np.append(self.data['fresh_frame'], fresh_frame_f1[i])
                self.data['fresh_times'] = np.append(self.data['fresh_times'], fresh_times_f1[i])
                self.data['type'].append(self.data['type'][f1_start_idx + i])

        for i in range(bb_num_f2):
            if i in chosen_idx:
                # chosen bb in f2, do not consider
                # print('object-' + str(chosen_ID[chosen_idx.index(i)]), end=' ')
                continue
            if score_f2[i] < 0.5:
                # print("The score of this detect bbox in next frame:", score_f2[i], "smaller than 0.5, continue.")
                continue

            # add new f2 bb.
            self.data['frame_ID'] = np.append(self.data['frame_ID'], frame_f2)
            # self.data['pc_center'] = np.append(self.data['pc_center'], [pc_center_f2], axis=0)
            self.data['start_frame'] = np.append(self.data['start_frame'], frame_f2)
            self.data['fresh_frame'] = np.append(self.data['fresh_frame'], frame_f2)
            self.data['fresh_times'] = np.append(self.data['fresh_times'], 0)
            current_object_num += 1
            self.data['max_object_num'] += 1
            self.data['object_ID'] = np.append(self.data['object_ID'], self.data['max_object_num'] - 1)
            self.data['lwh'] = np.append(self.data['lwh'], [size_f2[i]], axis=0)
            self.data['xyz'] = np.append(self.data['xyz'], [center_f2[i]], axis=0)
            self.data['yaw'] = np.append(self.data['yaw'], heading_angle_f2[i])
            self.data['score'] = np.append(self.data['score'], score_f2[i])
            self.data['type'].append(class2type[sem_class_f2[i]])

        # if frame_f2>50:
        #     IPython.embed()
        # print("Current frame has ", current_object_num, " objects.")
        self.data['last_object_num'] = current_object_num
        # print('now clean frame2 detected bboxes num is ', self.data['frame2_bboxes_clean_times'])

    def filter_object_dict(self):
        # judge if this object is the last frame to show
        self.data['object_ID'] = self.data['object_ID'].astype(int)
        self.data['start_frame'] = self.data['start_frame'].astype(int)
        self.data['fresh_frame'] = self.data['fresh_frame'].astype(int)
        self.data['frame_ID'] = self.data['frame_ID'].astype(int)

        # print("Before filter_object: self.data:", self.data)
        start_idx = self.data['start_frame']
        fresh_idx = self.data['fresh_frame']
        if start_idx.shape[0] != fresh_idx.shape[0]:
            assert True
        # object_compare = np.zeros((self.data['max_object_num'], 2))
        object_fresh_times = np.zeros(self.data['max_object_num'])
        for i in range(len(self)):
            # object_compare[self.data['object_ID'][i], 0] = self.data['start_frame'][i]
            # object_compare[self.data['object_ID'][i], 1] = self.data['fresh_frame'][i]
            # # last frame do not delete
            # if self.data['frame_ID'][i] == self.data['frame_ID'][len(self) - 1]:
            #     object_compare[self.data['object_ID'][i], 0] = -1

            object_fresh_times[self.data['object_ID'][i]] = self.data['fresh_times'][i]
        # print("object_compare: ", object_compare)
        # delete_object_id = np.where(object_compare[:, 0] == object_compare[:, 1])[0]

        # delete !!!!!!!!!!!
        delete_object_id = np.where(object_fresh_times <= 3)[0]

        # print("delete ", delete_object_id.shape[0], " and the deleted object_id: ", delete_object_id)
        # IPython.embed()
        i = 0
        while i < self.data['object_ID'].shape[0]:
            # print("i=", i)
            # print("self.data['object_ID'][i] ", self.data['object_ID'][i])
            # print("self.data['object_ID'] ", self.data['object_ID'])
            # print("delete_object_id", delete_object_id)
            if self.data['object_ID'][i] in delete_object_id:
                self.data['frame_ID'] = np.delete(self.data['frame_ID'], i, axis=0)
                # self.data['pc_center'] = np.append(self.data['pc_center'], [pc_center_f2], axis=0)
                self.data['start_frame'] = np.delete(self.data['start_frame'], i, axis=0)
                self.data['fresh_frame'] = np.delete(self.data['fresh_frame'], i, axis=0)
                self.data['object_ID'] = np.delete(self.data['object_ID'], i, axis=0)
                self.data['fresh_times'] = np.delete(self.data['fresh_times'], i, axis=0)
                self.data['lwh'] = np.delete(self.data['lwh'], i, axis=0)
                self.data['xyz'] = np.delete(self.data['xyz'], i, axis=0)
                self.data['yaw'] = np.delete(self.data['yaw'], i, axis=0)
                self.data['score'] = np.delete(self.data['score'], i, axis=0)
                del self.data['type'][i]
                i -= 1
            i += 1

        # filter all unfreshed frames in the last of the trajectories
        # IPython.embed()
        choosing_list = np.ones(self.data['max_object_num'])
        i = len(self) - 1
        while i >= 0:
            if self.data['frame_ID'][i] <= self.data['fresh_frame'][i]:
                choosing_list[self.data['object_ID'][i]] = 0
                i -= 1
                continue
            elif choosing_list[self.data['object_ID'][i]] == 1:

                # print('delete object', self.data['object_ID'][i], 'in frame ', self.data['frame_ID'][i])
                self.data['frame_ID'] = np.delete(self.data['frame_ID'], i, axis=0)
                # self.data['pc_center'] = np.append(self.data['pc_center'], [pc_center_f2], axis=0)
                self.data['start_frame'] = np.delete(self.data['start_frame'], i, axis=0)
                self.data['fresh_frame'] = np.delete(self.data['fresh_frame'], i, axis=0)
                self.data['object_ID'] = np.delete(self.data['object_ID'], i, axis=0)
                self.data['fresh_times'] = np.delete(self.data['fresh_times'], i, axis=0)
                self.data['lwh'] = np.delete(self.data['lwh'], i, axis=0)
                self.data['xyz'] = np.delete(self.data['xyz'], i, axis=0)
                self.data['yaw'] = np.delete(self.data['yaw'], i, axis=0)
                self.data['score'] = np.delete(self.data['score'], i, axis=0)
                del self.data['type'][i]
            i -= 1

    def save(self, calibfile, save_path):
        object_num = len(self)
        size = self.data['lwh']
        center = self.data['xyz']
        center_camera = transform_points(center, np.linalg.inv(self.data['last_T_cam_velo']))
        heading_angle = self.data['yaw']
        frame_id = self.data['frame_ID']
        object_id = self.data['object_ID']

        data, _ = get_calibration(calibfile)
        P = data['P2'].reshape((3, 4))
        if not size.shape[0] == center.shape[0] == heading_angle.shape[0] ==\
               frame_id.shape[0] == object_id.shape[0] == len(self.data['type']) == object_num:
            print("The object dict data is not correct. Please check the data.")
            assert True
        file = open(save_path, 'w')
        # file = open(os.path.join(result_save_path, str(EVAL_STEP_ * BATCH_SIZE + i).zfill(6)+'.txt'), 'a')
        for j in range(object_num):
            frame = frame_id[j]
            track_id = object_id[j]


            type = self.data['type'][j]
            trash1 = 0
            trash2 = 0
            beta = np.arctan(center_camera[j][0] / center_camera[j][2])
            alpha = ((-((np.pi + heading_angle[j]) + (np.pi + beta))) % (2 * np.pi)) - np.pi
            bbox_points = get_bboxes_points(size[j], heading_angle[j], center[j], self.data['last_T_cam_velo'])
            bbox_points_cam = transform_points(bbox_points, np.linalg.inv(self.data['last_T_cam_velo']))
            bbox_points_cam = np.append(bbox_points_cam, np.ones((8, 1)), axis=1)
            bbox_2d = np.transpose(np.dot(P, np.transpose(bbox_points_cam)))
            bbox_2d[:, 0] = bbox_2d[:, 0] / bbox_2d[:, 2]
            bbox_2d[:, 1] = bbox_2d[:, 1] / bbox_2d[:, 2]
            bbox_2d = bbox_2d[:, :2]
            # left = (bbox_2d[0, 0] + bbox_2d[1, 0]) / 2
            # top = (bbox_2d[0, 1] + bbox_2d[1, 1]) / 2
            # right = (bbox_2d[2, 0] + bbox_2d[3, 0]) / 2
            # bottom = (bbox_2d[4, 1] + bbox_2d[5, 1]) / 2
            # IPython.embed()
            left = np.min(bbox_2d[:, 0])
            if left < 0:
                # print("wrong left :", left)
                left = 0
                # continue
            bottom = np.max(bbox_2d[:, 1])
            if bottom > 370:
                # print("wrong bottom :", bottom)
                bottom = 370
                # continue
            right = np.max(bbox_2d[:, 0])
            if right > 1224:
                # print("wrong right :", right)
                right = 1224
                # continue
            top = np.min(bbox_2d[:, 1])
            if top < 0:
                # print("wrong top :", top)
                top = 0
                # continue
            dimensions = size[j]
            if dimensions[0] < 2:
                print("dimension 0 < 2: ", dimensions[0])
                continue
            elif dimensions[0] < 3:
                print("dimension 0 < 3 and > 2: ", dimensions[0])
                dimensions[0] = 3.3
            if np.any(dimensions < 0.01):
                print("have dimension < 0.01: ", dimensions)
                continue

            # print("final image bbox:", [left, top, right, bottom])

            location = center_camera[j] + [0, dimensions[2] / 2, 0]
            rotation_y = heading_angle[j]

            file.write(str(int(frame)))
            file.write(' ')
            file.write(str(int(track_id)))
            file.write(' ')
            file.write(type)
            file.write(' ')
            file.write(str(round(trash1, 6)))
            file.write(' ')
            file.write(str(round(trash2, 6)))
            file.write(' ')
            file.write(str(round(alpha, 6)))
            file.write(' ')
            file.write(str(round(left, 6)))
            file.write(' ')
            file.write(str(round(top, 6)))
            file.write(' ')
            file.write(str(round(right, 6)))
            file.write(' ')
            file.write(str(round(bottom, 6)))
            file.write(' ')
            file.write(str(round(dimensions[2], 6)))
            file.write(' ')
            file.write(str(round(dimensions[1], 6)))
            file.write(' ')
            file.write(str(round(dimensions[0], 6)))
            assert np.all(dimensions != 0)
            file.write(' ')
            file.write(str(round(location[0], 6)))
            file.write(' ')
            file.write(str(round(location[1], 6)))
            file.write(' ')
            file.write(str(round(location[2], 6)))
            file.write(' ')
            file.write(str(round(rotation_y, 6)))
            file.write('\n')
        file.close()
        # print('total clean frame2 detected bboxes num is ', self.data['frame2_bboxes_clean_times'])

    def visualize_dict(self, seq, frame_ID1, pc1, pc2, mask_f1, mask_f2, T_cam_velo, save_path, color_static=None, save_3d=False, is_test=False):
        if save_3d:
            pc1 = pc1[:, :3]
            pc2 = pc2[:, :3]
            pc = np.zeros((0, 3))
            color = np.zeros((0, 3))
            pc = np.append(pc, pc2, axis=0)
            color = np.append(color, np.array([[0, 0, 1]]) + np.zeros((pc2.shape[0], 3)), axis=0)
            frame_ID2 = frame_ID1+1
            for j in range(len(self)):
                if self.data['frame_ID'][j] == frame_ID2:
                    bbox_points = get_bboxes_points(self.data['lwh'][j], self.data['yaw'][j],
                                                             self.data['xyz'][j], self.data['last_T_cam_velo'])
                    bboxes_3d_points = draw_3d_bboxes(bbox_points)
                    pc = np.append(pc, bboxes_3d_points, axis=0)
                    color_i = [color_static[self.data['object_ID'][j]]] + np.zeros((bboxes_3d_points.shape[0], 3))
                    # print("in box, ", color_arr[self.data['object_ID'][j]])
                    color = np.append(color, color_i, axis=0)

            # draw trajectory
            color_idx = 0
            for j in range(len(self)):
                if self.data['frame_ID'][j] == frame_ID2:
                    start_frame = self.data['start_frame'][j]
                    object_id = self.data['object_ID'][j]
                    pos_traj = np.zeros((0, 3))
                    traj_length = 0
                    start_flag = 0
                    for trj in range(len(self)):
                        if self.data['frame_ID'][trj] >= start_frame and self.data['object_ID'][trj] == object_id:
                            pos_traj = np.append(pos_traj, [self.data['xyz'][trj]], axis=0)
                            traj_length += 1
                        if self.data['frame_ID'][trj] == frame_ID2 and self.data['object_ID'][trj] == object_id:
                            break
                    if traj_length == 1:
                        pc = np.append(pc, [self.data['xyz'][j]], axis=0)
                        color = np.append(color, [color_static[self.data['object_ID'][j]]], axis=0)
                    else:
                        for trj_id in range(traj_length - 1):
                            a = trj_id
                            b = a + 1
                            line_points = draw_line(pos_traj[a], pos_traj[b])
                            pc = np.append(pc, line_points, axis=0)
                            color_i = [color_static[self.data['object_ID'][j]]] + np.zeros((line_points.shape[0], 3))
                            color = np.append(color, color_i, axis=0)
                            # print("in tra, ", color_static[self.data['object_ID'][j]])
                    color_idx += 1
            pcview = o3d.geometry.PointCloud()
            pcview.points = o3d.utility.Vector3dVector(pc)
            pcview.colors = o3d.utility.Vector3dVector(color)
            save_path_pcd = save_path.replace('.png', '.pcd')
            o3d.io.write_point_cloud(save_path_pcd, pcview)
            print("Save 3D trajectory result into ", save_path_pcd)
        else:
            '''
            # red ^: past frame pred bounding boxes
            blue ^: current frame pred bounding boxes
            green : current ground truth bboxes
            trajectory: lines
            '''
            frame_ID2 = frame_ID1 + 1
            fig = plt.figure()
            axes = plt.gca()
            axes.set_aspect('equal')
            axes.set_xlim([0, 60])
            axes.set_ylim([-15, 15])
            colors = {'g': '#008000', 'r': '#FF0000', 'b': '#0000FF', 'm': '#FF00FF'}
            # draw point cloud
            # plt.scatter(pc1[:, 0], pc1[:, 1], marker='.', c=colors['r'], s=0.1)
            # plt.scatter(pc2[:, 0], pc2[:, 1], marker='.', c=colors['b'], s=0.1)

            foreground_pointcloud1_predict = pc1[np.where(mask_f1[..., 0] > mask_f1[..., 1])]
            foreground_pointcloud2_predict = pc2[np.where(mask_f2[..., 0] > mask_f2[..., 1])]
            background_pointcloud2_predict = pc2[np.where(mask_f2[..., 0] <= mask_f2[..., 1])]
            plt.scatter(foreground_pointcloud1_predict[:, 0], foreground_pointcloud1_predict[:, 1], marker='.', c=colors['r'], s=0.1, alpha=0.6)
            plt.scatter(foreground_pointcloud2_predict[:, 0], foreground_pointcloud2_predict[:, 1], marker='.', c=colors['b'], s=0.1, alpha=0.6)
            plt.scatter(background_pointcloud2_predict[:, 0], background_pointcloud2_predict[:, 1], marker='.',
                        c=colors['b'], s=0.1, alpha=0.1)

            if len(self) == 0:
                plt.savefig(save_path, dpi=800)
                plt.close('all')  # 关闭图 0
                return

            # draw dict bboxes
            dict_f1_num = 0
            dict_f2_num = 0
            # IPython.embed()
            for j in range(len(self)):
                # if self.data['frame_ID'][j] == frame_ID1:
                #     # print("frame_ID1", j)
                #     # print("lwh", self.data['lwh'][j])
                #     bboxes_birdeye_points = draw_birdeye_bbox_shc(self.data['lwh'][j], self.data['yaw'][j],
                #                                                   self.data['xyz'][j], self.data['last_T_cam_velo'])
                #     # IPython.embed()
                #     plt.scatter(bboxes_birdeye_points[:, 0], bboxes_birdeye_points[:, 1], marker='^', c=colors['r'], s=0.005)
                #     plt.text(self.data['xyz'][j][0], self.data['xyz'][j][1], str(round(self.data['score'][j], 2)),
                #              ha="center", va="center", color=colors['r'], size=4)
                #     dict_f1_num += 1
                if self.data['frame_ID'][j] == frame_ID2:
                    bboxes_birdeye_points = draw_birdeye_bbox_shc(self.data['lwh'][j], self.data['yaw'][j],
                                                                  self.data['xyz'][j], self.data['last_T_cam_velo'])
                    plt.scatter(bboxes_birdeye_points[:, 0], bboxes_birdeye_points[:, 1], marker='^', c=colors['r'], s=0.005)
                    plt.text(self.data['xyz'][j][0], self.data['xyz'][j][1], str(round(self.data['score'][j], 2)),
                             ha="center", va="center", color=colors['r'], size=5)
                    dict_f2_num += 1
            # print("in dict, there are ", dict_f2_num, " in frame 2")

            if not is_test:
                # draw label
                label_f1_num = 0
                label_f2_num = 0
                # bboxes_label_birdeye_points = np.zeros((0, 3))
                # for j in range(batch_bboxes_labels_f2['bboxes_num']):
                #     if np.sum(batch_bboxes_labels_f2['bboxes_size'][i][j]) == 0:
                #         break
                #     bboxes_label_birdeye_points = np.append(bboxes_label_birdeye_points,
                #                                             draw_birdeye_bbox_shc(
                #                                                 batch_bboxes_labels_f2['bboxes_size'][i][j],
                #                                                 batch_bboxes_labels_f2['heading_angles'][i][j],
                #                                                 batch_bboxes_labels_f2['bboxes_xyz'][i][j],
                #                                                 self.data['last_T_cam_velo']), axis=0)
                #     label_f1_num += 1
                # plt.scatter(bboxes_label_birdeye_points[:, 0], bboxes_label_birdeye_points[:, 1], marker='x', c=colors['r'], s=0.005)

                # bboxes_label_birdeye_points = np.zeros((0, 3))
                # for j in range(batch_bboxes_labels_f2['bboxes_num']):
                #     if np.sum(batch_bboxes_labels_f2['bboxes_size'][i][j]) == 0:
                #         break
                #     bboxes_label_birdeye_points = np.append(bboxes_label_birdeye_points,
                #                                             draw_birdeye_bbox_shc(
                #                                                 batch_bboxes_labels_f2['bboxes_size'][i][j],
                #                                                 batch_bboxes_labels_f2['heading_angles'][i][j],
                #                                                 batch_bboxes_labels_f2['bboxes_xyz'][i][j] + batch_bboxes_labels_f2['pc_center'][i],
                #                                                 self.data['last_T_cam_velo']), axis=0)
                #     label_f2_num += 1

                label_dir = '/data/KITTI_object_tracking/training/label_02'
                bbox3d_label = get_label_bbox3d(label_dir, seq, frame_ID2)
                size_label = bbox3d_label[:, 5:2:-1]
                heading_label = bbox3d_label[:, -1]
                center_label = bbox3d_label[:, :3]
                center_label = transform_points(center_label, T_cam_velo)
                label_bboxes_points = np.zeros((0, 3))
                for i in range(size_label.shape[0]):
                    label_bboxes_points = np.append(label_bboxes_points,
                                                    get_bboxes_points(size_label[i], heading_label[i], center_label[i],
                                                                      T_cam_velo), axis=0)
                label_bboxes_birdeye_points = draw_birdeye_bboxes(label_bboxes_points)
                plt.scatter(label_bboxes_birdeye_points[:, 0], label_bboxes_birdeye_points[:, 1], c=colors['g'], s=0.005)
                # print("in label, there are ", label_f2_num, " in frame 2")

            # draw trajectory
            for j in range(len(self)):
                if self.data['frame_ID'][j] == frame_ID2:
                    start_frame = self.data['start_frame'][j]
                    object_id = self.data['object_ID'][j]
                    pos_traj = np.zeros((0, 3))
                    traj_idx = 0
                    start_flag = 0
                    for trj in range(len(self)):
                        if self.data['frame_ID'][trj] >= start_frame and self.data['object_ID'][trj] == object_id:
                            pos_traj = np.append(pos_traj, [self.data['xyz'][trj]], axis=0)
                            traj_idx += 1
                        if self.data['frame_ID'][trj] == frame_ID2 and self.data['object_ID'][trj] == object_id:
                            plt.text(self.data['xyz'][trj][0], self.data['xyz'][trj][1] + 1,
                                     self.data['type'][trj] + " " + str(int(self.data['object_ID'][trj])), ha="center", va="center", size=7)
                    if traj_idx == 1:
                        plt.plot(self.data['xyz'][j][0], self.data['xyz'][j][1])
                    else:
                        plt.plot(pos_traj[:, 0], pos_traj[:, 1])
                    # elif traj_idx == 2:
                    #     plt.plot(pos_traj[:, 0], pos_traj[:, 1])
                    # else:
                    #     x = pos_traj[:, 0]
                    #     y = pos_traj[:, 1]
                    #     xnew = np.linspace(x.min(), x.max(), 20)  # 300 represents number of points to make between x.min and x.max
                    #     y_smooth = spline(x, y, xnew)
                    #     plt.plot(xnew, y_smooth)

            if not is_test:
                # draw label trajectory
                label_file = os.path.join(label_dir, '%04d.txt' % seq)
                lines = [line.rstrip() for line in open(label_file)]
                # label = []
                object_id_gt = []
                for l in range(len(lines)):
                    frame_i = int(lines[l].split(' ')[0])
                    label_i = dataset_utils.read_line(lines[l])
                    if frame_i == frame_ID2:
                        # label.append(label_i)
                        object_id_gt.append(label_i['track_id'])
                for id in range(len(object_id_gt)):
                    id_this = object_id_gt[id]
                    pos_traj_i = np.zeros((0, 3))
                    traj_idx = 0
                    for l in range(len(lines)):
                        frame_i = int(lines[l].split(' ')[0])
                        if frame_i < self.data['frame_ID'][0]:
                            continue
                        if frame_i > frame_ID2:
                            break
                        label_i = dataset_utils.read_line(lines[l])
                        if label_i['track_id'] == id_this:
                            # pos_cam = np.array(label_i['t'])
                            xyz = np.array(label_i['t'])
                            xyz_center_cam = xyz - [0, label_i['h'] / 2, 0]
                            xyz_center = np.dot(self.data['last_T_cam_velo'], np.append(xyz_center_cam, 1))[:3]
                            # pos_lidar = np.dot(self.data['last_T_cam_velo'], np.append(pos_cam, 1))[:3] + batch_bboxes_labels_f2['pc_center'][i]
                            pos_lidar = xyz_center # + batch_bboxes_labels_f2['pc_center'][i]
                            pos_traj_i = np.append(pos_traj_i, [pos_lidar], axis=0)
                            traj_idx += 1
                    plt.plot(pos_traj_i[:, 0], pos_traj_i[:, 1], ':')

            plt.savefig(save_path, dpi=800)
            plt.close('all')  # 关闭图 0


    def visualize_dict_after_filter(self, frame_ID2, pc2, bboxes_label_f2, dict_save_path, label_file, color_arr, save_3d=False, is_test=False):
        if len(self) == 0:
            return

        if save_3d:
            pc = np.zeros((0, 3))
            color = np.zeros((0, 3))
            pc = np.append(pc, pc2, axis=0)
            color = np.append(color, np.array([[0, 0, 1]]) + np.zeros((pc2.shape[0], 3)), axis=0)

            for j in range(len(self)):
                if self.data['frame_ID'][j] == frame_ID2:
                    bbox_points = get_bboxes_points(self.data['lwh'][j], self.data['yaw'][j],
                                                             self.data['xyz'][j], self.data['last_T_cam_velo'])
                    bboxes_3d_points = draw_3d_bboxes(bbox_points)
                    pc = np.append(pc, bboxes_3d_points, axis=0)
                    color_i = [color_arr[self.data['object_ID'][j]]] + np.zeros((bboxes_3d_points.shape[0], 3))
                    # print("in box, ", color_arr[self.data['object_ID'][j]])
                    color = np.append(color, color_i, axis=0)

            # draw trajectory
            color_idx = 0
            for j in range(len(self)):
                if self.data['frame_ID'][j] == frame_ID2:
                    start_frame = self.data['start_frame'][j]
                    object_id = self.data['object_ID'][j]
                    pos_traj = np.zeros((0, 3))
                    traj_length = 0
                    start_flag = 0
                    for trj in range(len(self)):
                        if self.data['frame_ID'][trj] >= start_frame and self.data['object_ID'][trj] == object_id:
                            pos_traj = np.append(pos_traj, [self.data['xyz'][trj]], axis=0)
                            traj_length += 1
                        if self.data['frame_ID'][trj] == frame_ID2 and self.data['object_ID'][trj] == object_id:
                            break
                    if traj_length == 1:
                        pc = np.append(pc, [self.data['xyz'][j]], axis=0)
                        color = np.append(color, [color_arr[self.data['object_ID'][j]]], axis=0)
                    else:
                        for trj_id in range(traj_length - 1):
                            a = trj_id
                            b = a + 1
                            line_points = draw_line(pos_traj[a], pos_traj[b])
                            pc = np.append(pc, line_points, axis=0)
                            color_i = [color_arr[self.data['object_ID'][j]]] + np.zeros((line_points.shape[0], 3))
                            color = np.append(color, color_i, axis=0)
                            # print("in tra, ", color_arr[self.data['object_ID'][j]])
                    color_idx += 1
            pcview = o3d.geometry.PointCloud()
            pcview.points = o3d.utility.Vector3dVector(pc)
            pcview.colors = o3d.utility.Vector3dVector(color)
            o3d.io.write_point_cloud(dict_save_path, pcview)
            print("Save 3D trajectory result into ", dict_save_path)
        else:
            fig = plt.figure()
            axes = plt.gca()
            axes.set_aspect('equal')
            axes.set_xlim([0, 60])
            axes.set_ylim([-15, 15])
            colors = {'g': '#008000', 'r': '#FF0000', 'b': '#0000FF', 'm': '#FF00FF'}

            dict_f2_num = 0
            for j in range(len(self)):
                if self.data['frame_ID'][j] == frame_ID2:
                    bboxes_birdeye_points = draw_birdeye_bbox_shc(self.data['lwh'][j], self.data['yaw'][j],
                                                                  self.data['xyz'][j], self.data['last_T_cam_velo'])
                    plt.scatter(bboxes_birdeye_points[:, 0], bboxes_birdeye_points[:, 1], marker='^', c=colors['b'],
                                s=0.005)
                    plt.text(self.data['xyz'][j][0], self.data['xyz'][j][1], str(round(self.data['score'][j], 2)),
                             ha="center", va="center", color=colors['b'], size=5)
                    dict_f2_num += 1

            if not is_test:
                # draw label
                label_f2_num = 0
                bboxes_label_birdeye_points = np.zeros((0, 3))
                for j in range(bboxes_label_f2['bboxes_num']):
                    bboxes_label_birdeye_points = np.append(bboxes_label_birdeye_points,
                                                            draw_birdeye_bbox_shc(
                                                                bboxes_label_f2['bboxes_lwh'][j],
                                                                bboxes_label_f2['heading_angle'][j],
                                                                bboxes_label_f2['bboxes_xyz'][j] +
                                                                bboxes_label_f2['pc_center'],
                                                                self.data['last_T_cam_velo']), axis=0)
                    label_f2_num += 1
                plt.scatter(bboxes_label_birdeye_points[:, 0], bboxes_label_birdeye_points[:, 1], c=colors['g'], s=0.005)
                # print("in frame ", frame_ID2, ", there are ", label_f2_num, " in gt and ", dict_f2_num, " in detection.")

            # draw trajectory
            for j in range(len(self)):
                if self.data['frame_ID'][j] == frame_ID2:
                    start_frame = self.data['start_frame'][j]
                    object_id = self.data['object_ID'][j]
                    pos_traj = np.zeros((0, 3))
                    traj_idx = 0
                    start_flag = 0
                    for trj in range(len(self)):
                        if self.data['frame_ID'][trj] >= start_frame and self.data['object_ID'][trj] == object_id:
                            pos_traj = np.append(pos_traj, [self.data['xyz'][trj]], axis=0)
                            traj_idx += 1
                        if self.data['frame_ID'][trj] == frame_ID2 and self.data['object_ID'][trj] == object_id:
                            plt.text(self.data['xyz'][trj][0], self.data['xyz'][trj][1] + 1,
                                     self.data['type'][trj] + " " + str(int(self.data['object_ID'][trj])), ha="center",
                                     va="center", size=5)
                            break
                    if traj_idx == 1:
                        plt.plot(self.data['xyz'][j][0], self.data['xyz'][j][1])
                    else:
                        plt.plot(pos_traj[:, 0], pos_traj[:, 1])
                    # elif traj_idx == 2:
                    #     plt.plot(pos_traj[:, 0], pos_traj[:, 1])
                    # else:
                    #     x = pos_traj[:, 0]
                    #     y = pos_traj[:, 1]
                    #     xnew = np.linspace(x.min(), x.max(), 20)  # 300 represents number of points to make between x.min and x.max
                    #     y_smooth = spline(x, y, xnew)
                    #     plt.plot(xnew, y_smooth)

            if not is_test:
                # draw label trajectory
                lines = [line.rstrip() for line in open(label_file)]
                # label = []
                object_id_gt = []
                for l in range(len(lines)):
                    frame_i = int(lines[l].split(' ')[0])
                    label_i = dataset_utils.read_line(lines[l])
                    if frame_i == frame_ID2 and label_i['type'] in type_whitelist:
                        # label.append(label_i)
                        object_id_gt.append(label_i['track_id'])
                for id in range(len(object_id_gt)):
                    id_this = object_id_gt[id]
                    pos_traj_i = np.zeros((0, 3))
                    traj_idx = 0
                    for l in range(len(lines)):
                        frame_i = int(lines[l].split(' ')[0])
                        if frame_i < self.data['frame_ID'][0]:
                            continue
                        if frame_i > frame_ID2:
                            break
                        label_i = dataset_utils.read_line(lines[l])
                        if label_i['track_id'] == id_this:
                            # pos_cam = np.array(label_i['t'])
                            xyz = np.array(label_i['t'])
                            xyz_center_cam = xyz - [0, label_i['h'] / 2, 0]
                            xyz_center = np.dot(self.data['last_T_cam_velo'], np.append(xyz_center_cam, 1))[:3]
                            # pos_lidar = np.dot(self.data['last_T_cam_velo'], np.append(pos_cam, 1))[:3] + batch_bboxes_labels_f2['pc_center'][i]
                            pos_lidar = xyz_center  # + batch_bboxes_labels_f2['pc_center'][i]
                            pos_traj_i = np.append(pos_traj_i, [pos_lidar], axis=0)
                            traj_idx += 1
                    plt.plot(pos_traj_i[:, 0], pos_traj_i[:, 1], ':')

            plt.scatter(pc2[:, 0], pc2[:, 1], marker='.', c=colors['b'], s=0.1, alpha=0.1)

            plt.savefig(dict_save_path, dpi=800)
            print("Save birdseye-view trajectory result into ", dict_save_path)
            plt.close('all')  # 关闭图 0