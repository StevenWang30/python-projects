import sys
sys.path.append("..")
from point_cloud_compression.rangeimage_to_pointcloud import rangeimage_to_pointcloud
import glob
import numpy as np
import IPython
import open3d as o3d

lidar_angular_xy_range_ = 360
max_lidar_angular_z_ = 2
min_lidar_angular_z_ = -24.5
range_x_ = 64
range_y_ = 2000
nearest_bound_ = 0.5
furthest_bound_ = 120
if_show_ground_ = True

files = glob.glob('./data/txt_data/*.txt')
for f in files:
    rangeimage = np.loadtxt(f)
    print(f)
    pc = rangeimage_to_pointcloud(rangeimage, lidar_angular_xy_range_, max_lidar_angular_z_, min_lidar_angular_z_,
                             nearest_bound_, furthest_bound_)

    pc_vis = o3d.geometry.PointCloud()
    pc_vis.points = o3d.utility.Vector3dVector(pc)
    o3d.visualization.draw_geometries([pc_vis])
    point_cloud_save_dir = f.replace('/txt_data/', '/pointcloud_data/').replace('.txt', '.pcd')
    o3d.io.write_point_cloud(point_cloud_save_dir, pc_vis)
