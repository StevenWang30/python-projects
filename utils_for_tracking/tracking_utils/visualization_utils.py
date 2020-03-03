import matplotlib.pyplot as plt
from utils_for_tracking.KITTI_dataset_utils.utils import *
from utils_for_tracking.KITTI_dataset_utils.dataset import *
from utils_for_tracking.tracking_utils.box_util import *

def save_for_tracking_visulization(pc1, pc2, tracking_pred, tracking_label, pc1_chosen_i, pc1_tracked_gt,
                    mask_f1, size_f1, heading_f1, center_f1, scores_f1,
                    mask_f2, size_f2, heading_f2, center_f2, scores_f2, T_cam_velo, save_path):
    bb_num_f1 = size_f1.shape[0]
    mask_pos = np.where(mask_f1[..., 0] > mask_f1[..., 1])
    # tracking_pred = tracking_pred[mask_pos]
    # score_f1 = mask_f1[mask_pos][:, 0]
    pc1_foreground = pc1[mask_pos]
    pc2_foreground = pc2[np.where(mask_f2[..., 0] > mask_f2[..., 1])]
    pc2_background = pc2[np.where(mask_f2[..., 0] <= mask_f2[..., 1])]
    pc1_tracked = pc1_chosen_i + tracking_pred

    tracked_bboxes_points_f1 = np.zeros((0, 3))
    bboxes_points_f1 = np.zeros((0, 3))
    if bb_num_f1 > 0:
        for bb in range(bb_num_f1):

            idx = find_points_idx_in_bbox(pc1_chosen_i, center_f1[bb], size_f1[bb], heading_f1[bb], T_cam_velo)

            # if sum(idx.astype(np.float32)) < 50:
            #     continue

            tracking_bb_this = tracking_pred[idx]
            # score_f1_this = score_f1[idx]
            if tracking_bb_this.shape[0] > 0:
                # order_f1 = score_f1_this.argsort()[::-1]
                # tracking_bb_this = tracking_bb_this[order_f1[:order_f1.shape[0] // 2]]
                tracking_mean = np.mean(tracking_bb_this, 0)
            else:
                tracking_mean = np.zeros(7)
                # print('no points in this bbox')
            center_f1_tracked = center_f1[bb] + tracking_mean[:3]
            # size_f1[bb] += tracking_mean[3:6]
            # heading_f1[bb] += tracking_mean[6]
            tracked_bboxes_points_i = get_bboxes_points(size_f1[bb], heading_f1[bb], center_f1_tracked, T_cam_velo)
            tracked_bboxes_points_i = draw_birdeye_bboxes(tracked_bboxes_points_i)
            tracked_bboxes_points_f1 = np.append(tracked_bboxes_points_f1, tracked_bboxes_points_i, axis=0)

            bboxes_points_i = get_bboxes_points(size_f1[bb], heading_f1[bb], center_f1[bb], T_cam_velo)
            bboxes_points_i = draw_birdeye_bboxes(bboxes_points_i)
            bboxes_points_f1 = np.append(bboxes_points_f1, bboxes_points_i, axis=0)

    gt_tracked_bboxes_points_f1 = np.zeros((0, 3))
    if bb_num_f1 > 0:
        for bb in range(bb_num_f1):
            idx = find_points_idx_in_bbox(pc1, center_f1[bb], size_f1[bb], heading_f1[bb], T_cam_velo)
            tracking_bb_this = tracking_label[idx]
            # score_f1_this = score_f1[idx]
            if tracking_bb_this.shape[0] > 0:
                # order_f1 = score_f1_this.argsort()[::-1]
                # tracking_bb_this = tracking_bb_this[order_f1[:order_f1.shape[0] // 2]]
                tracking_mean = np.mean(tracking_bb_this, 0)
            else:
                tracking_mean = np.zeros(7)
                # print('no points in this bbox')
            center_f1_tracked_gt = center_f1[bb] + tracking_mean[:3]
            # size_f1[bb] += tracking_mean[3:6]
            # heading_f1[bb] += tracking_mean[6]
            gt_tracked_bboxes_points_i = get_bboxes_points(size_f1[bb], heading_f1[bb], center_f1_tracked_gt, T_cam_velo)
            gt_tracked_bboxes_points_i = draw_birdeye_bboxes(gt_tracked_bboxes_points_i)
            gt_tracked_bboxes_points_f1 = np.append(gt_tracked_bboxes_points_f1, gt_tracked_bboxes_points_i, axis=0)

    # draw pred bboxes in frame2
    bb_num_f2 = size_f2.shape[0]
    bboxes_points_f2 = np.zeros((0, 3))
    for j in range(bb_num_f2):
        bboxes_points_f2 = np.append(bboxes_points_f2,
                                                draw_birdeye_bbox_shc(
                                                    size_f2[j],
                                                    heading_f2[j],
                                                    center_f2[j],
                                                    T_cam_velo), axis=0)

    # draw bboxes in frame1
    # draw tracked bboxes
    # draw label bboxes in frame2
    # draw pc2
    fig = plt.figure()
    colors = {'g': '#008000', 'r': '#FF0000', 'b': '#0000FF', 'm': '#FF00FF'}
    plt.axis('off')
    # plt.scatter(pc2_foreground[:, 0], pc2_foreground[:, 1], c=colors['b'], s=0.1)
    # plt.scatter(pc1_foreground[:, 0], pc1_foreground[:, 1], c=colors['r'], s=0.1)
    # plt.scatter(pc2_background[:, 0], pc2_background[:, 1], c=colors['b'], alpha=0.1, s=0.1)
    # plt.scatter(pc1_tracked[:, 0], pc1_tracked[:, 1], c=colors['g'], s=0.1, alpha=0.3)
    # plt.scatter(pc1_tracked_gt[:, 0], pc1_tracked_gt[:, 1], c=colors['m'], s=0.1, alpha=0.1)
    # # plt.scatter(pc1_chosen_i[:, 0], pc1_chosen_i[:, 1], c=colors['m'], s=0.1)

    plt.scatter(bboxes_points_f1[:, 0], bboxes_points_f1[:, 1], c=colors['r'], s=0.005, alpha=0.5)
    plt.scatter(tracked_bboxes_points_f1[:, 0], tracked_bboxes_points_f1[:, 1], c=colors['g'], s=0.01, alpha=0.5)
    plt.scatter(bboxes_points_f2[:, 0], bboxes_points_f2[:, 1], c=colors['b'], s=0.005, alpha=0.5)
    plt.scatter(gt_tracked_bboxes_points_f1[:, 0], gt_tracked_bboxes_points_f1[:, 1], c=colors['m'], s=0.005, alpha=0.5)
    plt.savefig(save_path, dpi=800)
    plt.close('all')  # 关闭图 0

def draw_line(point1, point2):
    split_num = 100
    dxyz = point1 - point2
    dxyz_reso = dxyz / split_num
    line_points = [point2]
    for i in range(split_num):
        point_i = point2 + i * dxyz_reso
        line_points = np.append(line_points, [point_i], axis=0)
    return line_points

def draw_birdeye_bboxes(bboxes_points):
    birdeye_bbox_points = np.zeros((0, 3))
    for i in range(bboxes_points.shape[0] // 8):
        bbox_points = bboxes_points[i * 8: i * 8 + 8]
        z_mean = np.mean(bbox_points[..., 2])
        bbox_birdeye = bbox_points[np.where(bbox_points[..., 2] > z_mean)]
        birdeye_bbox_points = np.append(birdeye_bbox_points, draw_line(bbox_birdeye[0], bbox_birdeye[1]), axis=0)
        birdeye_bbox_points = np.append(birdeye_bbox_points, draw_line(bbox_birdeye[0], bbox_birdeye[3]), axis=0)
        birdeye_bbox_points = np.append(birdeye_bbox_points, draw_line(bbox_birdeye[2], bbox_birdeye[1]), axis=0)
        birdeye_bbox_points = np.append(birdeye_bbox_points, draw_line(bbox_birdeye[2], bbox_birdeye[3]), axis=0)

    return birdeye_bbox_points

def draw_3d_bboxes(bboxes_points):
    bbox_points = np.zeros((0, 3))
    for i in range(bboxes_points.shape[0] // 8):
        bbox_points_8 = bboxes_points[i * 8: i * 8 + 8]
        bbox_points = np.append(bbox_points, draw_line(bbox_points_8[0], bbox_points_8[1]), axis=0)
        bbox_points = np.append(bbox_points, draw_line(bbox_points_8[0], bbox_points_8[3]), axis=0)
        bbox_points = np.append(bbox_points, draw_line(bbox_points_8[0], bbox_points_8[4]), axis=0)
        bbox_points = np.append(bbox_points, draw_line(bbox_points_8[2], bbox_points_8[6]), axis=0)
        bbox_points = np.append(bbox_points, draw_line(bbox_points_8[2], bbox_points_8[1]), axis=0)
        bbox_points = np.append(bbox_points, draw_line(bbox_points_8[2], bbox_points_8[3]), axis=0)
        bbox_points = np.append(bbox_points, draw_line(bbox_points_8[7], bbox_points_8[6]), axis=0)
        bbox_points = np.append(bbox_points, draw_line(bbox_points_8[7], bbox_points_8[4]), axis=0)
        bbox_points = np.append(bbox_points, draw_line(bbox_points_8[7], bbox_points_8[3]), axis=0)
        bbox_points = np.append(bbox_points, draw_line(bbox_points_8[5], bbox_points_8[4]), axis=0)
        bbox_points = np.append(bbox_points, draw_line(bbox_points_8[5], bbox_points_8[6]), axis=0)
        bbox_points = np.append(bbox_points, draw_line(bbox_points_8[5], bbox_points_8[1]), axis=0)

    return bbox_points

def draw_birdeye_bbox_shc(size_pred, heading_pred, center_pred, T_cam_velo):  # _sch means size heading center
    bbox_points = get_bboxes_points(size_pred, heading_pred, center_pred, T_cam_velo)
    z_mean = np.mean(bbox_points[..., 2])
    bbox_birdeye = bbox_points[np.where(bbox_points[..., 2] > z_mean)]
    birdeye_bbox_points = draw_line(bbox_birdeye[0], bbox_birdeye[1])
    birdeye_bbox_points = np.append(birdeye_bbox_points, draw_line(bbox_birdeye[0], bbox_birdeye[3]), axis=0)
    birdeye_bbox_points = np.append(birdeye_bbox_points, draw_line(bbox_birdeye[2], bbox_birdeye[1]), axis=0)
    birdeye_bbox_points = np.append(birdeye_bbox_points, draw_line(bbox_birdeye[2], bbox_birdeye[3]), axis=0)

    return birdeye_bbox_points

def get_bboxes_points(size_pred, heading_pred, center_pred, T_cam_velo):
    [l, w, h] = size_pred
    angle = heading_pred
    # print('angle = ', angle)
    # print('T_cam_velo = ', T_cam_velo)
    # angle = 0
    bbox_points_cam = bbox(l, w, h, roty_angle=angle)
    bbox_points = transform_points(bbox_points_cam, T_cam_velo)
    offset = center_pred
    # offset = proposals_xyz
    bbox_points = bbox_points + offset
    return bbox_points

def get_NMS_bboxes_points(score, size_pred, heading_pred, center_pred, T_cam_velo, nms_th=0.05):
    order = score.argsort()[::-1]

    # '''sample the score as 0.9 rate'''
    # order = order[:math.ceil(order.shape[0]*0.9)]
    bboxes_points = np.zeros((0, 3))
    while order.size > 0:
        if size_pred[order[0]][0] < 2 or np.any(size_pred[order[0]] < 0.01):
            order = order[1:]
            continue
        bbox_points_maxscore = get_bboxes_points(size_pred[order[0]], heading_pred[order[0]], center_pred[order[0]],
                                                 T_cam_velo)
        bboxes_points = np.append(bboxes_points, bbox_points_maxscore, axis=0)
        bbox_points_a = transform_points(bbox_points_maxscore, np.linalg.inv(T_cam_velo))
        order_max = order[0]
        order = order[1:]
        iou_all = np.zeros((order.shape[0]))
        for i in range(order.shape[0]):
            # speed up
            if size_pred[order[i]][0] < 2 or np.any(size_pred[order[i]] < 0.01):
                iou_all[i] = 1
                continue
            center_dis = math.sqrt(sum((center_pred[order[i]] - center_pred[order_max]) * (
                        center_pred[order[i]] - center_pred[order_max])))
            if center_dis < min(size_pred[order_max]):
                iou_all[i] = 1
                continue
            if center_dis > max(size_pred[order_max]):
                iou_all[i] = 0
                continue
            bbox_points_this = get_bboxes_points(size_pred[order[i]], heading_pred[order[i]], center_pred[order[i]],
                                                 T_cam_velo)
            # IPython.embed()
            bbox_points_b = transform_points(bbox_points_this, np.linalg.inv(T_cam_velo))
            ''' compare the iou2d, not iou3d'''
            iou3d, iou_all[i] = box3d_iou(bbox_points_a, bbox_points_b)
            # iou_all[i], iou2d = box3d_iou(bbox_points_a, bbox_points_b)
        inds = np.where(iou_all <= nms_th)[0]
        order = order[inds]
        # print(order)
    return bboxes_points

def get_NMS_bboxes_labels(score, size_pred, heading_pred, center_pred, class_scores_pred, T_cam_velo, nms_th=0.05):
    order = score.argsort()[::-1]
    # '''sample the score as 0.9 rate'''
    # order = order[:math.ceil(order.shape[0]*0.9)]
    semantic_classes = np.argmax(class_scores_pred, axis=1)
    bboxes_points = np.zeros((0, 3))
    size_output = np.zeros((0, 3))
    heading_output = np.zeros((0, 1))
    class_output = np.zeros((0, 1))
    scores_output = np.zeros((0, 1))
    center_output = np.zeros((0, 3))
    while order.size > 0:
        if size_pred[order[0]][0] < 2 or np.any(size_pred[order[0]] < 0.01):
            order = order[1:]
            continue
        bbox_points_maxscore = get_bboxes_points(size_pred[order[0]], heading_pred[order[0]], center_pred[order[0]],
                                                 T_cam_velo)
        bbox_points_a = transform_points(bbox_points_maxscore, np.linalg.inv(T_cam_velo))
        bboxes_points = np.append(bboxes_points, bbox_points_maxscore, axis=0)
        size_output = np.append(size_output, [size_pred[order[0]]], axis=0)
        heading_output = np.append(heading_output, heading_pred[order[0]])
        center_output = np.append(center_output, [center_pred[order[0]]], axis=0)
        class_output = np.append(class_output, semantic_classes[order[0]])
        scores_output = np.append(scores_output, score[order[0]])
        order_max = order[0]
        order = order[1:]
        iou_all = np.zeros((order.shape[0]))
        for i in range(order.shape[0]):
            # speed up
            if size_pred[order[i]][0] < 2 or np.any(size_pred[order[i]] < 0.01):
                iou_all[i] = 1
                continue
            center_dis = math.sqrt(sum((center_pred[order[i]] - center_pred[order_max]) * (
                        center_pred[order[i]] - center_pred[order_max])))
            if center_dis < min(size_pred[order_max]):
                iou_all[i] = 1
                continue
            if center_dis > max(size_pred[order_max]):
                iou_all[i] = 0
                continue
            bbox_points_this = get_bboxes_points(size_pred[order[i]], heading_pred[order[i]], center_pred[order[i]],
                                                 T_cam_velo)
            # IPython.embed()
            bbox_points_b = transform_points(bbox_points_this, np.linalg.inv(T_cam_velo))
            ''' compare the iou2d, not iou3d'''
            iou3d, iou_all[i] = box3d_iou(bbox_points_a, bbox_points_b)
            # iou_all[i], iou2d = box3d_iou(bbox_points_a, bbox_points_b)
        inds = np.where(iou_all <= nms_th)[0]
        order = order[inds]
        # print(order)
    return bboxes_points, size_output, heading_output, center_output, class_output, scores_output

def optimize_NMS_bboxes(score, size_pred, heading_pred, center_pred, class_scores_pred,
                        NMS_size, NMS_heading, NMS_center, T_cam_velo):
    # bboxes_points = np.zeros((0, 3))
    size_output = np.zeros((0, 3))
    heading_output = np.zeros((0, 1))
    class_output_n = np.zeros((0, 1))
    scores_output = np.zeros((0, 1))
    center_output = np.zeros((0, 3))
    bb_num = NMS_size.shape[0]
    THRESHOLD = [0.5, 0.5, 0.6]
    for bb in range(bb_num):
        [l, w, h] = NMS_size[bb] + THRESHOLD
        pointcloud_dif = center_pred - NMS_center[bb]
        pointcloud_trans = np.transpose(np.dot(roty4(-1 * NMS_heading[bb]), np.dot(np.linalg.inv(T_cam_velo),
                                                                                   np.transpose(
                                                                                       np.append(pointcloud_dif,
                                                                                                 np.ones((
                                                                                                         pointcloud_dif.shape[
                                                                                                             0],
                                                                                                         1)),
                                                                                                 axis=1)))))[:, :3]
        idx_x = np.logical_and(pointcloud_trans[:, 0] <= l / 2.0, pointcloud_trans[:, 0] >= -l / 2.0)
        idx_y = np.logical_and(pointcloud_trans[:, 1] <= w / 2.0, pointcloud_trans[:, 1] >= -w / 2.0)
        idx_z = np.logical_and(pointcloud_trans[:, 2] <= h / 2.0, pointcloud_trans[:, 2] >= -h / 2.0)
        idx = np.logical_and(idx_x, np.logical_and(idx_y, idx_z))
        center_in_box = center_pred[idx]
        size_in_box = size_pred[idx]
        heading_in_box = heading_pred[idx]
        score_in_box = score[idx]
        class_in_box = class_scores_pred[idx]
        order = score_in_box.argsort()[::-1]
        if order.shape[0] // 5 > 4:
            point_num = 4
        else:
            point_num = math.ceil(order.shape[0] / 5)
        order_filtered = []
        for i in range(point_num):
            if size_in_box[order[i]][0] < 2 or np.any(size_in_box[order[i]] < 0.01):
                continue
            order_filtered.append(order[i])
        if len(order_filtered) == 0:
            continue
        elif len(order_filtered) <= 2:
            # to 324 0.708 0.773 0.712
            center_in_box_mean = np.mean(center_in_box[order_filtered[:point_num]], 0)
            size_in_box_mean = np.mean(size_in_box[order_filtered[:point_num]], 0)
            heading_in_box_mean = np.mean(heading_in_box[order_filtered[:point_num]], 0)
            class_in_box_mean = np.mean(class_in_box[order_filtered[:point_num]], 0)
            semantic_classes = np.argmax(class_in_box_mean)
            score_in_box_mean = np.mean(score_in_box[order_filtered[:point_num]], 0)
            # bbox_points_mean = get_bboxes_points(size_in_box_mean, heading_in_box_mean, center_in_box_mean, T_cam_velo)
        else:
            center_in_box_mean = (sum(center_in_box[order_filtered[:point_num]], 0) - np.max(
                center_in_box[order_filtered[:point_num]], 0) - np.min(center_in_box[order_filtered[:point_num]],
                                                                       0)) / (len(order_filtered) - 2)
            size_in_box_mean = (sum(size_in_box[order_filtered[:point_num]], 0) - np.max(
                size_in_box[order_filtered[:point_num]], 0) - np.min(size_in_box[order_filtered[:point_num]],
                                                                     0)) / (len(order_filtered) - 2)
            heading_in_box_mean = (sum(heading_in_box[order_filtered[:point_num]]) - np.max(
                heading_in_box[order_filtered[:point_num]]) - np.min(
                heading_in_box[order_filtered[:point_num]])) / (len(order_filtered) - 2)
            class_in_box_mean = (sum(class_in_box[order_filtered[:point_num]], 0) - np.max(
                class_in_box[order_filtered[:point_num]], 0) - np.min(class_in_box[order_filtered[:point_num]],
                                                                      0)) / (len(order_filtered) - 2)
            semantic_classes = np.argmax(class_in_box_mean)
            score_in_box_mean = (sum(score_in_box[order_filtered[:point_num]]) - np.max(
                score_in_box[order_filtered[:point_num]]) - np.min(score_in_box[order_filtered[:point_num]])) / (
                                            len(order_filtered) - 2)
            # bbox_points_mean = get_bboxes_points(size_in_box_mean, heading_in_box_mean, center_in_box_mean, T_cam_velo)

        # bboxes_points = np.append(bboxes_points, bbox_points_mean, axis=0)
        scores_output = np.append(scores_output, score_in_box_mean)
        size_output = np.append(size_output, [size_in_box_mean], axis=0)
        heading_output = np.append(heading_output, heading_in_box_mean)
        center_output = np.append(center_output, [center_in_box_mean], axis=0)
        class_output_n = np.append(class_output_n, semantic_classes)

    # # do not change heading angle
    # heading_output = NMS_heading

    class_output = np.zeros((class_output_n.shape[0], NUM_SIZE_CLUSTER))
    class_output[:, class_output_n.astype(int)] = 1
    # re NMS, for the optimizer will lead to connect
    bboxes_points, size_output, heading_output, center_output, class_output, scores_output = \
        get_NMS_bboxes_labels(scores_output, size_output, heading_output, center_output, class_output, T_cam_velo)
    return bboxes_points, size_output, heading_output, center_output, class_output, scores_output
