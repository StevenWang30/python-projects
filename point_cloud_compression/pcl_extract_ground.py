import numpy as np
from utils.utils import load_bin
import IPython
import os
import open3d as o3d
import pcl
import math

def pcl_extract_ground(cloud):

    # fil = cloud.make_passthrough_filter()
    # fil.set_filter_field_name("z")
    # fil.set_filter_limits(0, 1.5)
    # cloud_filtered = fil.filter()
    seg = cloud.make_segmenter_normals(ksearch=10)
    seg.set_optimize_coefficients(True)
    seg.set_model_type(pcl.SACMODEL_NORMAL_PLANE)
    seg.set_normal_distance_weight(0.1)
    seg.set_method_type(pcl.SAC_RANSAC)
    seg.set_max_iterations(100)
    seg.set_distance_threshold(0.4)
    indices, model = seg.segment()
    cloud_ground = cloud.extract(indices, negative=False)
    cloud_deground = cloud.extract(indices, negative=True)
    point_cloud_deground = np.asarray(cloud_deground)
    point_cloud_ground = np.asarray(cloud_ground)
    # ground_save_path = "/home/skwang/data/2011_09_26/2011_09_26_drive_0009_extract/velodyne_points_ground_extractioin/ground"
    # deground_save_path = "/home/skwang/data/2011_09_26/2011_09_26_drive_0009_extract/velodyne_points_ground_extractioin/point_cloud"
    # pcl.save(cloud_ground, os.path.join(ground_save_path, data_name))
    # pcl.save(cloud_deground, os.path.join(deground_save_path, data_name+'.pcd'))
    # print(model)

    # # ground_center = np.mean(point_cloud_ground, axis=0)
    # # pcground = point_cloud_ground - ground_center
    # IPython.embed()
    # # u, s, vh = np.linalg.svd(pcground[:500,:], full_matrices=True)
    # # model = vh[2,:]
    # # d = -1*(model[0]*ground_center[0]+model[1]*ground_center[1]+model[2]*ground_center[2])
    # # model = np.append(model,d)
    # # print(model)
    # model = np.multiply(-1, model)
    # draw_extraction_result(point_cloud_ground, point_cloud_deground)
    # IPython.point_cloud_degroundembed()
    return point_cloud_deground, point_cloud_ground, model

def save_ground(ground_model, data_name, deground_save_path):
    if not os.path.isdir(deground_save_path):
        os.makedirs(deground_save_path)
        print("Make dir to " + deground_save_path)
    np.savetxt(os.path.join(deground_save_path, data_name + ".txt"), ground_model)


def draw_extraction_result(ground, deground):
    ground_temp = o3d.geometry.PointCloud()
    ground_temp.points = o3d.utility.Vector3dVector(ground)
    deground_temp = o3d.geometry.PointCloud()
    deground_temp.points = o3d.utility.Vector3dVector(deground)
    ground_temp.paint_uniform_color([1, 0, 0])
    deground_temp.paint_uniform_color([0, 0, 1])
    o3d.visualization.draw_geometries([ground_temp, deground_temp])


def visual_deground_result(pointcloud_path, ground_path):
    [a, b, c, d] = np.loadtxt(ground_path)
    # print(a, b, c, d)
    pointcloud = load_bin(pointcloud_path)
    ground = []
    deground = []
    threshold = 0.4
    for i in range(pointcloud.shape[0]):
        x = pointcloud[i][0]
        y = pointcloud[i][1]
        z = pointcloud[i][2]
        dis = abs(a * x + b * y + c * z + d) / math.sqrt(a * a + b * b + c * c)
        if dis < threshold:
            ground.append([x, y, z])
        else:
            deground.append([x, y, z])
    ground_temp = o3d.geometry.PointCloud()
    ground_temp.points = o3d.utility.Vector3dVector(ground)
    deground_temp = o3d.geometry.PointCloud()
    deground_temp.points = o3d.utility.Vector3dVector(deground)
    ground_temp.paint_uniform_color([1, 0, 0])
    deground_temp.paint_uniform_color([0, 0, 1])
    o3d.visualization.draw_geometries([ground_temp, deground_temp])

if __name__ == "__main__":
    data_path = "/home/skwang/data/KITTI_object_tracking/training/velodyne"
    files = os.listdir(data_path)
    files.sort(key=lambda x: int(x.split('.')[0]))
    for dir in files:
        dir_path = os.path.join(data_path, dir)
        bin_files = os.listdir(dir_path)
        bin_files.sort(key=lambda x: int(x.split('.')[0]))
        for i in bin_files:
            data_name = i.split(".")[0]
            print("compose " + os.path.join(dir_path, data_name + '.bin') + " file.")
            cloud = load_bin(os.path.join(dir_path, data_name + '.bin'))
            cloud = pcl.PointCloud(cloud)

            cloud_filtered, ground, ground_model = pcl_extract_ground(cloud)

            deground_save_path = dir_path.replace("velodyne", "ground")
            save_ground(ground_model, data_name, deground_save_path)