import open3d as o3d
import os
import numpy as np
import IPython
import math
from pointcloud_to_rangeimage import *
from PIL import Image

lidar_angular_xy_range_ = 360
max_lidar_angular_z_ = 2
min_lidar_angular_z_ = -24.5
range_x_ = 64
range_y_ = 2000
nearest_bound_ = 0.5
furthest_bound_ = 120
if_show_ground_ = True


KITTI_raw_data_path = '/data/KITTI_rawdata'
sequence_name = '2011_09_26_drive_0009_extract'
save_root = '/data/FutureGAN_data/KITTI_resized_128x128'

# read point cloud
vel_path = os.path.join(KITTI_raw_data_path, sequence_name, "velodyne_points/data")
test_datapath = []
files = os.listdir(vel_path)
files.sort(key=lambda x: int(x.split('.')[0]))
for txtfile in files:
    if txtfile.split('.')[-1] != 'txt':
        continue
    file_name = txtfile.split('.')[0]
    pointcloud_path = os.path.join(vel_path, txtfile)
    pointcloud = np.zeros((0, 3))
    with open(pointcloud_path, 'r') as f:
        for line in f.readlines():
            line = line.rstrip()
            point = [float(i) for i in line.split(' ')[:3]]
            pointcloud = np.append(pointcloud, [point], axis=0)

    range_image_array = pointcloud_to_rangeimage(pointcloud, lidar_angular_xy_range_, max_lidar_angular_z_,
                                                 min_lidar_angular_z_, range_x_, range_y_)

    range_image_array = remove_balck_line_and_remote_points(range_image_array)