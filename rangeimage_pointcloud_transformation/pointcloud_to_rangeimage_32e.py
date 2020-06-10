import sys
sys.path.append("..")
from point_cloud_compression.pointcloud_to_rangeimage import pointcloud_to_rangeimage
import glob
import numpy as np
import IPython
import open3d as o3d
import os

lidar_angular_xy_range_ = 360
max_lidar_angular_z_ = 10.68
min_lidar_angular_z_ = -30.68
range_x_ = 32
range_y_ = 2000  #should be 1600
nearest_bound_ = 0.5
furthest_bound_ = 120
if_show_ground_ = True

files = glob.glob('/data/rangeimage_prediction_32E/pointcloud_txt_file/autoware_32e/*.txt')
files.sort()
for f in files:
    pointcloud = np.loadtxt(f)
    print(f)
    ri = pointcloud_to_rangeimage(pointcloud, lidar_angular_xy_range_, max_lidar_angular_z_, min_lidar_angular_z_,
                             range_x_, range_y_)
    if '/pointcloud_txt_file/' in f:
        range_image_save_path = f.replace('/pointcloud_txt_file/', '/rangeimage_txt_file/')
    else:
        assert True
    os.makedirs(range_image_save_path.replace(range_image_save_path.split('/')[-1], ''), exist_ok=True)
    np.savetxt(range_image_save_path, ri, fmt='%.5f')