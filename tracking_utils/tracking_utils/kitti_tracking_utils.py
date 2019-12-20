import numpy as np
import os
# import det3.methods.second_tracking.tracking_utils.kitti_utils as kitti_utils
import IPython
# from lib.tracking_utils.tf_image_vis import get_bboxes_points
import matplotlib.pyplot as plt
# import lib.tracking_utils.dataset as dataset

def save_kitti_tracking_format_test_detection(sample_id, calib, bbox3d, kitti_output_dir, scores, img_shape, threshold):
    seq = int(sample_id.split('-')[0])
    idx_frame1 = int(sample_id.split(',')[0].split('-')[1])
    idx_frame2 = int(sample_id.split(',')[1].split('-')[1])

    # save_dir = os.path.join(kitti_output_dir, '%04d' % seq)
    # os.makedirs(save_dir, exist_ok=True)

    scores = scores[:, 0]
    scores = sigmoid(scores)

    corners3d = kitti_utils.boxes3d_to_corners3d(bbox3d)
    img_boxes, _ = calib.corners3d_to_img_boxes(corners3d)

    img_boxes[:, 0] = np.clip(img_boxes[:, 0], 0, img_shape[1] - 1)
    img_boxes[:, 1] = np.clip(img_boxes[:, 1], 0, img_shape[0] - 1)
    img_boxes[:, 2] = np.clip(img_boxes[:, 2], 0, img_shape[1] - 1)
    img_boxes[:, 3] = np.clip(img_boxes[:, 3], 0, img_shape[0] - 1)

    img_boxes_w = img_boxes[:, 2] - img_boxes[:, 0]
    img_boxes_h = img_boxes[:, 3] - img_boxes[:, 1]
    box_valid_mask = np.logical_and(img_boxes_w < img_shape[1] * 0.8, img_boxes_h < img_shape[0] * 0.8)

    kitti_output_file = os.path.join(kitti_output_dir, '%04d.txt' % seq)
    object_id = 0
    with open(kitti_output_file, 'a+') as f:
        for k in range(bbox3d.shape[0]):
            if box_valid_mask[k] == 0:
                continue
            x, z, ry = bbox3d[k, 0], bbox3d[k, 2], bbox3d[k, 6]
            beta = np.arctan2(z, x)
            alpha = -np.sign(beta) * np.pi / 2 + beta + ry

            # print('%d %d %s -1 -1 %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f' %
            #       (idx_frame1, object_id, cfg.CLASSES, alpha, img_boxes[k, 0], img_boxes[k, 1], img_boxes[k, 2], img_boxes[k, 3],
            #        bbox3d[k, 3], bbox3d[k, 4], bbox3d[k, 5], bbox3d[k, 0], bbox3d[k, 1], bbox3d[k, 2],
            #        bbox3d[k, 6], scores[k]), file=f)
            if scores[k] >= threshold:
                print('%d %d %s -1 -1 %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f' %
                      (idx_frame1, object_id, "Car", alpha, img_boxes[k, 0], img_boxes[k, 1], img_boxes[k, 2],
                       img_boxes[k, 3],
                       bbox3d[k, 3], bbox3d[k, 4], bbox3d[k, 5], bbox3d[k, 0], bbox3d[k, 1], bbox3d[k, 2],
                       bbox3d[k, 6]), file=f)

                object_id += 1


def visualize_detection_result(pointcloud, mask, sample_id, calib, bbox3d, kitti_output_dir, scores, img_shape, threshold):
    root = '/data/KITTI_object_tracking/results_PointRCNNTrackNet/detection_visulization_new'

    size_output = bbox3d[:, 5:2:-1]
    heading_output = bbox3d[:, -1]
    center_output = bbox3d[:, :3]
    # scores = scores[:, 0]
    # scores = sigmoid(scores)

    pos_idx = np.where(scores > threshold)

    size_output = size_output[pos_idx]
    heading_output = heading_output[pos_idx]
    center_output = center_output[pos_idx]
    scores = scores[pos_idx]

    T_cam_velo = np.linalg.inv( np.append(calib.V2C, [[0,0,0,1]], axis=0) )
    T_velo_cam = np.append(calib.V2C, [[0,0,0,1]], axis=0)

    pointcloud = transform_points(pointcloud, T_cam_velo)
    center_output = transform_points(center_output, T_cam_velo)


    # draw predicted bboxes:
    mask_positive_idx = np.where(mask > 0)
    mask_negative_idx = np.where(mask == 0)
    foreground_pointcloud_predict = pointcloud[mask_positive_idx]
    background_pointcloud_predict = pointcloud[mask_negative_idx]


    NMS_bboxes_points = np.zeros((0, 3))
    for i in range(size_output.shape[0]):
        NMS_bboxes_points = np.append(NMS_bboxes_points, get_bboxes_points(size_output[i], heading_output[i], center_output[i], T_cam_velo), axis=0)
    NMS_bboxes_birdeye_points = draw_birdeye_bboxes(NMS_bboxes_points)

    label_dir = '/data/KITTI_object_tracking/training/label_02'
    seq = int(sample_id.split(',')[0].split('-')[0])
    frame1 = int(sample_id.split(',')[0].split('-')[1])
    bbox3d_label = get_label_bbox3d(label_dir, seq, frame1)
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
    # bboxes_label_birdeye_points = np.zeros((0, 3))
    # for j in range(batch_bboxes_label_in['bboxes_num']):
    #     if np.sum(batch_bboxes_label_in['bboxes_size'][i][j]) == 0:
    #         break
    #     bboxes_label_birdeye_points = np.append(bboxes_label_birdeye_points,
    #                               draw_birdeye_bbox_shc(batch_bboxes_label_in['bboxes_size'][i][j], batch_bboxes_label_in['heading_angles'][i][j], batch_bboxes_label_in['bboxes_xyz'][i][j], T_cam_velo),
    #                               axis=0)

    dir_name = os.path.join(root, sample_id.split(',')[0].split('-')[0])
    if not os.path.isdir(dir_name): os.makedirs(dir_name)
    file_name_this = sample_id.split(',')[0].split('-')[1] + '_detection.png'
    save_path = os.path.join(dir_name, file_name_this)
    # save_path = os.path.join(vis_save_path, str(EVAL_STEP_ * BATCH_SIZE + i).zfill(6)+'.png')
    fig = plt.figure()
    axes = plt.gca()
    axes.set_aspect('equal')
    axes.set_xlim([0, 60])
    axes.set_ylim([-15, 15])
    colors = {'g': '#008000', 'r': '#FF0000', 'b': '#0000FF', 'm': '#FF00FF'}
    plt.scatter(foreground_pointcloud_predict[:, 0], foreground_pointcloud_predict[:, 1], c=colors['g'], s=0.2)
    plt.scatter(background_pointcloud_predict[:, 0], background_pointcloud_predict[:, 1], c=colors['r'], s=0.2)
    plt.scatter(NMS_bboxes_birdeye_points[:, 0], NMS_bboxes_birdeye_points[:, 1], c=colors['b'], s=0.2)
    plt.scatter(label_bboxes_birdeye_points[:, 0], label_bboxes_birdeye_points[:, 1], c=colors['m'], s=0.2)
    for bb_idx in range(size_output.shape[0]):
        # center = np.mean(NMS_bboxes_points[bb_idx*8:bb_idx*8+8], 0)
        plt.text(center_output[bb_idx][0], center_output[bb_idx][1], str(round(scores[bb_idx],2)), ha="center", va="center", size=6)
        plt.plot(center_output[bb_idx][0], center_output[bb_idx][1], '.')

    plt.savefig(save_path, dpi=500)
    plt.close('all')  # 关闭图 0

    # print(center_output)
    # IPython.embed()


    # size_label = batch_bboxes_label_in['bboxes_size'][i]
    # bb = size_label.shape[0]
    # for tt in range(size_label.shape[0]):
    #     if np.all(size_label[tt] == 0):
    #         bb = tt
    #         break
    # size_label = size_label[:bb]
    # heading_label = batch_bboxes_label_in['heading_angles'][i][:bb]
    # center_label = batch_bboxes_label_in['bboxes_xyz'][i][:bb]
    # score_label = np.ones(bb)
    # NMS_bboxes_label_points = get_NMS_bboxes_points(score_label, size_label, heading_label, center_label, T_cam_velo)
    # NMS_bboxes_label_3d_points = draw_3d_bboxes(NMS_bboxes_label_points)
    # # save to pcd
    # p1 = foreground_pointcloud_predict
    # c1 = np.array([[0, 1, 0]]) + np.zeros((p1.shape[0], 3))
    # p2 = background_pointcloud_predict
    # c2 = np.array([[1, 0, 0]]) + np.zeros((p2.shape[0], 3))
    # p3 = NMS_bboxes_3d_points
    # c3 = np.array([[0, 0, 1]]) + np.zeros((p3.shape[0], 3))
    # p4 = NMS_bboxes_label_3d_points
    # c4 = np.array([[1, 0.5, 0.4]]) + np.zeros((p4.shape[0], 3))
    #
    # pc = np.append(p3, p4, axis=0)
    # pc = np.append(pc, p1, axis=0)
    # pc = np.append(pc, p2, axis=0)
    # color = np.append(c3, c4, axis=0)
    # color = np.append(color, c1, axis=0)
    # color = np.append(color, c2, axis=0)
    #
    # pcview = o3d.geometry.PointCloud()
    # pcview.points = o3d.utility.Vector3dVector(pc)
    # pcview.colors = o3d.utility.Vector3dVector(color)
    # o3d.io.write_point_cloud(save_path.replace('.png', '.pcd'), pcview)


def delete_all_dir_file(path):
    if not os.path.isdir(path):
        os.makedirs(path)
        print("Make dir to " + path)
    else:
        filelist = os.listdir(path)  # 列出该目录下的所有文件名
        for f in filelist:
            filepath = os.path.join(path, f)  # 将文件名映射成绝对路劲
            if os.path.isfile(filepath):  # 判断该文件是否为文件或者文件夹
                os.remove(filepath)  # 若为文件，则直接删除
                print(str(filepath) + " removed!")
            # else:
            #     # 若为folder
            #     shutil.rmtree(filepath)
            #     print(str(filepath) + " removed!")
        print("remove all old files in " + path)

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


def transform_points(points, Tr):
    return np.dot(Tr, np.append(points, np.ones((points.shape[0], 1)), axis=1).transpose()).transpose()[:, :-1]

def bbox(l, w, h, roty_angle=None):
    x_corners = [l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2]
    y_corners = [h / 2, h / 2, h / 2, h / 2, -h / 2, -h / 2, -h / 2, -h / 2]
    z_corners = [w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2]
    corners_3d = np.vstack([x_corners, y_corners, z_corners])
    if roty_angle:
        corners_3d = roty(roty_angle) @ corners_3d
    return np.transpose(corners_3d)


def draw_birdeye_bboxes(bboxes_points):
    birdeye_bbox_points = np.zeros((0, 3))
    for i in range(bboxes_points.shape[0] // 8):
        bbox_points = bboxes_points[i * 8 : i * 8 + 8]
        z_mean = np.mean(bbox_points[..., 2])
        bbox_birdeye = bbox_points[np.where(bbox_points[..., 2] > z_mean)]
        birdeye_bbox_points = np.append(birdeye_bbox_points, draw_line(bbox_birdeye[0], bbox_birdeye[1]), axis=0)
        birdeye_bbox_points = np.append(birdeye_bbox_points, draw_line(bbox_birdeye[0], bbox_birdeye[3]), axis=0)
        birdeye_bbox_points = np.append(birdeye_bbox_points, draw_line(bbox_birdeye[2], bbox_birdeye[1]), axis=0)
        birdeye_bbox_points = np.append(birdeye_bbox_points, draw_line(bbox_birdeye[2], bbox_birdeye[3]), axis=0)

    return birdeye_bbox_points



def roty(t):
    """Rotation about the y-axis."""
    c = np.cos(t)
    s = np.sin(t)
    return np.array([[c, 0, s],
                     [0, 1, 0],
                     [-s, 0, c]])


def roty4(t):
    """Rotation about the y-axis."""
    c = np.cos(t)
    s = np.sin(t)
    return np.array([[c, 0, s, 0],
                     [0, 1, 0, 0],
                     [-s, 0, c, 0],
                     [0, 0, 0, 1]])


def draw_line(point1, point2):
    split_num = 100
    dxyz = point1 - point2
    dxyz_reso = dxyz / split_num
    line_points = [point2]
    for i in range(split_num):
        point_i = point2 + i * dxyz_reso
        line_points = np.append(line_points, [point_i], axis=0)
    return line_points


def sigmoid(inx):
    return 1 / (1 + np.exp(-inx))

def get_label_bbox3d(dir, seq, frame):
    file_name = os.path.join(dir, '%04d.txt' % seq)
    lines = [line.rstrip() for line in open(file_name)]
    bbox3d = np.zeros((0, 7))
    for l in range(len(lines)):
        frame_i = int(lines[l].split(' ')[0])
        if frame_i < frame:
            continue
        if frame_i > frame:
            break
        if lines[l].split(' ')[2] == 'Car' or lines[l].split(' ')[2] == 'Van':
            bbox3d = np.append(bbox3d, [[float(lines[l].split(' ')[3 + 10]), float(lines[l].split(' ')[4 + 10]),
                                         float(lines[l].split(' ')[5 + 10]), float(lines[l].split(' ')[0 + 10]),
                                         float(lines[l].split(' ')[1 + 10]), float(lines[l].split(' ')[2 + 10]),
                                         float(lines[l].split(' ')[6 + 10])]], axis=0)
    return bbox3d


def draw_birdeye_bbox_shc(size_pred, heading_pred, center_pred, T_cam_velo): # _sch means size heading center
    bbox_points = get_bboxes_points(size_pred, heading_pred, center_pred, T_cam_velo)
    z_mean = np.mean(bbox_points[..., 2])
    bbox_birdeye = bbox_points[np.where(bbox_points[..., 2] > z_mean)]
    birdeye_bbox_points = draw_line(bbox_birdeye[0], bbox_birdeye[1])
    birdeye_bbox_points = np.append(birdeye_bbox_points, draw_line(bbox_birdeye[0], bbox_birdeye[3]), axis=0)
    birdeye_bbox_points = np.append(birdeye_bbox_points, draw_line(bbox_birdeye[2], bbox_birdeye[1]), axis=0)
    birdeye_bbox_points = np.append(birdeye_bbox_points, draw_line(bbox_birdeye[2], bbox_birdeye[3]), axis=0)

    return birdeye_bbox_points

def draw_3d_bboxes(bboxes_points):
    bbox_points = np.zeros((0, 3))
    for i in range(bboxes_points.shape[0] // 8):
        bbox_points_8 = bboxes_points[i * 8 : i * 8 + 8]
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