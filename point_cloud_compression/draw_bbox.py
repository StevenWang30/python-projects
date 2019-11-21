import numpy as np
# from utils.box_util import box3d_iou
# from utils.utils import roty4
# from utils.utils import find_points_idx_in_bbox
import math
import IPython


def draw_bbox_of_pc_cluster(pc_cluster):
    # bbox_center = np.mean(pc_cluster, 0)
    bbox_center = (np.max(pc_cluster, 0) + np.min(pc_cluster, 0)) / 2
    print(bbox_center[2])
    bbox_size = np.max(pc_cluster, 0) - np.min(pc_cluster, 0)
    [l, h, w] = bbox_size
    x_corners = [l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2]
    y_corners = [h / 2, h / 2, h / 2, h / 2, -h / 2, -h / 2, -h / 2, -h / 2]
    z_corners = [w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2]
    corners_3d = np.vstack([x_corners, y_corners, z_corners])
    corners_3d = np.transpose(corners_3d) + bbox_center
    bbox_points = draw_3d_bbox(corners_3d)
    return bbox_points


def draw_3d_bbox(bbox_vertices):
    bbox_points = np.zeros((0, 3))
    bbox_points = np.append(bbox_points, draw_line(bbox_vertices[0], bbox_vertices[1]), axis=0)
    bbox_points = np.append(bbox_points, draw_line(bbox_vertices[0], bbox_vertices[3]), axis=0)
    bbox_points = np.append(bbox_points, draw_line(bbox_vertices[0], bbox_vertices[4]), axis=0)
    bbox_points = np.append(bbox_points, draw_line(bbox_vertices[2], bbox_vertices[6]), axis=0)
    bbox_points = np.append(bbox_points, draw_line(bbox_vertices[2], bbox_vertices[1]), axis=0)
    bbox_points = np.append(bbox_points, draw_line(bbox_vertices[2], bbox_vertices[3]), axis=0)
    bbox_points = np.append(bbox_points, draw_line(bbox_vertices[7], bbox_vertices[6]), axis=0)
    bbox_points = np.append(bbox_points, draw_line(bbox_vertices[7], bbox_vertices[4]), axis=0)
    bbox_points = np.append(bbox_points, draw_line(bbox_vertices[7], bbox_vertices[3]), axis=0)
    bbox_points = np.append(bbox_points, draw_line(bbox_vertices[5], bbox_vertices[4]), axis=0)
    bbox_points = np.append(bbox_points, draw_line(bbox_vertices[5], bbox_vertices[6]), axis=0)
    bbox_points = np.append(bbox_points, draw_line(bbox_vertices[5], bbox_vertices[1]), axis=0)
    return bbox_points


def draw_line(point1, point2):
    split_num = 50
    dxyz = point1 - point2
    dxyz_reso = dxyz / split_num
    line_points = [point2]
    for i in range(split_num):
        point_i = point2 + i * dxyz_reso
        line_points = np.append(line_points, [point_i], axis=0)
    return line_points
