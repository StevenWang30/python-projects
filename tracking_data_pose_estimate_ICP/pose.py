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
from utils import load_bin, draw_result
import copy
import scipy.io as sio

data_root = "/data/KITTI_object_tracking/training"
data_velodyne_root = data_root + '/velodyne'
files = os.listdir(data_velodyne_root)
files.sort(key=lambda x: int(x))
for dir in files:
    dir_path = os.path.join(data_velodyne_root, dir)
    bin_files = os.listdir(dir_path)
    bin_files.sort(key=lambda x: int(x.split('.')[0]))
    pose_total = np.zeros((0, 3, 4))
    for i in range(len(bin_files) - 1):
        fn1 = bin_files[i]
        f1 = os.path.join(data_velodyne_root, dir, fn1)
        fn2 = bin_files[i + 1]
        f2 = os.path.join(data_velodyne_root, dir, fn2)
        pc1 = load_bin(f1)
        pc1_o3d = o3d.geometry.PointCloud()
        pc1_o3d.points = o3d.utility.Vector3dVector(pc1)
        pc2 = load_bin(f2)
        pc2_o3d = o3d.geometry.PointCloud()
        pc2_o3d.points = o3d.utility.Vector3dVector(pc2)
        trans_init = np.asarray([[1.0, 0.0, 0.0, 0.0],
                                 [0.0, 1.0, 0.0, 0.0],
                                 [0.0, 0.0, 1.0, 0.0],
                                 [0.0, 0.0, 0.0, 1.0]])
        T1, pc1_trans = icp(pc1_o3d, pc2_o3d, trans_init)
        print("seq: ", dir, ", ",  fn1 + "-" + fn2 + ", pose is:\n", T1)
        pose_total = np.append(pose_total, [T1[:3, :]], 0)
    if int(dir) == 1:
        IPython.embed()
    pose_total = np.einsum('kli->lik', pose_total)
    save_path = os.path.join(data_root, 'pose', dir + '.mat')
    sio.savemat(save_path, {'pose': pose_total})
#
# seq = '0000'
# f1_name = '000000.bin'
# f1 = os.path.join(data_root, 'velodyne', seq, f1_name)
# f2_name = '000001.bin'
# f2 = os.path.join(data_root, 'velodyne', seq, f2_name)
# pc1 = load_bin(f1)
# pc1_o3d = o3d.geometry.PointCloud()
# pc1_o3d.points = o3d.utility.Vector3dVector(pc1)
# pc2 = load_bin(f2)
# pc2_o3d = o3d.geometry.PointCloud()
# pc2_o3d.points = o3d.utility.Vector3dVector(pc2)
#
# # pc1_o3d = o3d.io.read_point_cloud(f1)
# # pc2_o3d = o3d.io.read_point_cloud(f2)
# # pc1 = np.asarray(pc1_o3d.points)
# # pc2 = np.asarray(pc2_o3d.points)
# # ground_truth = load_groundtruth_pose(pc2_frame, pc3_frame)
#
# # test icp
# trans_init = np.asarray([[1.0, 0.0, 0.0, 0.0],
#                          [0.0, 1.0, 0.0, 0.0],
#                          [0.0, 0.0, 1.0, 0.0],
#                          [0.0, 0.0, 0.0, 1.0]])
# T1, pc1_trans = icp(pc1_o3d, pc2_o3d, trans_init)
# draw_result(pc1_o3d, pc2_o3d, T1, "result/result.pcd")
#
# print("T1 is:")
# print(T1)
# print("")
#
#
#
# save_path = os.path.join(data_root, 'pose', seq + '.mat')
# sio.savemat(save_path, {'pose': T1})

