import sys
sys.path.append("..")
from point_cloud_compression.pointcloud_to_rangeimage import pointcloud_to_rangeimage
import glob
import numpy as np
import IPython
import open3d as o3d
import os

lidar_angular_xy_range_ = 360
max_lidar_angular_z_ = 15.1
min_lidar_angular_z_ = -15.1
range_x_ = 16
range_y_ = 1000
nearest_bound_ = 0.5
furthest_bound_ = 120
if_show_ground_ = True

files = glob.glob('/data/rangeimage_prediction_VLP16/outdoor/scans/*.txt')
files.sort()
range_image_save_root = '/data/rangeimage_prediction_VLP16/rangeimage_txt_file/outdoor'
os.makedirs(range_image_save_root, exist_ok=True)

def load_bin(file_name):
    scan = np.fromfile(file_name, dtype=np.float32)
    scan = scan.reshape((-1, 4))
    point_cloud_array = scan[:, :3]
    # pc = o3d.geometry.PointCloud()
    # pc.points = o3d.utility.Vector3dVector(point_cloud_array)
    return point_cloud_array


i = 0
for f in files:
    if i >= 30:
        break
    # pointcloud = load_bin(f)
    pointcloud = np.loadtxt(f, skiprows=1)
    pointcloud = pointcloud[:, :3]

    # pc_vis = o3d.geometry.PointCloud()
    # pc_vis.points = o3d.utility.Vector3dVector(pointcloud)
    # o3d.visualization.draw_geometries([pc_vis])

    print(f)
    ri = pointcloud_to_rangeimage(pointcloud, lidar_angular_xy_range_, max_lidar_angular_z_, min_lidar_angular_z_,
                             range_x_, range_y_)
    save_path = os.path.join(range_image_save_root, str(i) + '.txt')
    np.savetxt(save_path, ri, fmt='%.5f')
    i += 1