import math
import numpy as np
import os
import sys
import IPython
import open3d as o3d
from PIL import Image
import matplotlib.pyplot as plt
import time
from icp_python import icp_open3d as icp
import copy


def load_bin(file_name):
    scan = np.fromfile(file_name, dtype=np.float32)
    scan = scan.reshape((-1, 4))
    point_cloud_array = scan[:, :3]
    # pc = o3d.geometry.PointCloud()
    # pc.points = o3d.utility.Vector3dVector(point_cloud_array)
    return point_cloud_array


def draw_result(source, target, transformation, pcd_save_name):
    # source = o3d.geometry.PointCloud()
    # source.points = o3d.utility.Vector3dVector(source_array)
    # target = o3d.geometry.PointCloud()
    # target.points = o3d.utility.Vector3dVector(target_array)
    transformed = copy.deepcopy(source)
    transformed.transform(transformation)

    source.paint_uniform_color([0, 0, 1]) # blue
    target.paint_uniform_color([0, 1, 0]) # green
    transformed.paint_uniform_color([1, 0, 0]) # red

    # o3d.visualization.draw_geometries([source, target, transformed])

    save = copy.deepcopy(source)
    points = np.append(np.asarray(source.points), np.asarray(target.points), axis=0)
    points = np.append(points, np.asarray(transformed.points), axis=0)
    save.points = o3d.utility.Vector3dVector(points)
    colors = np.append(np.asarray(source.colors), np.asarray(target.colors), axis=0)
    colors = np.append(colors, np.asarray(transformed.colors), axis=0)
    save.colors = o3d.utility.Vector3dVector(colors)

    o3d.io.write_point_cloud(pcd_save_name, save)
