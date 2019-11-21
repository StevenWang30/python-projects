import os
import open3d as o3d
from pointcloud_to_rangeimage import *
import pcl
import IPython
from pcl_extract_ground import pcl_extract_ground
from utils.utils import *
from draw_bbox import draw_bbox_of_pc_cluster
from pcl_filters import statistical_outlier_removal_filter
from PIL import Image
# import scipy.misc
import imageio

data_dir = './example_data/raw_point_cloud.pcd'
result_root = './results'
filtered_point_cloud_save_dir = os.path.join(result_root, "filtered_original_point_cloud.pcd")
extract_ground_point_cloud_save_dir = os.path.join(result_root, "ground_extracted_point_cloud.pcd")
extract_ground_clustered_point_cloud_save_dir = os.path.join(result_root, "ec_point_cloud_after_ground_extraction.pcd")

pc_o3d = o3d.io.read_point_cloud(data_dir)
# o3d.visualization.draw_geometries([pc_o3d])
pc_nparray = np.asarray(pc_o3d.points).astype(np.float32)

# Extract ground
pc_pcl = pcl.PointCloud(pc_nparray)

# # radius outlier removal
# radius = 0.5
# min_point_num = 5
# pc_filtered_pcl = radius_outlier_removal_pcl(radius, min_point_num, pc_pcl)

# StatisticalOutlierRemovalFilter
mean_k = 50
pc_filtered_pcl = statistical_outlier_removal_filter(mean_k, pc_nparray)
pc_filtered_nparray = np.asarray(pc_filtered_pcl)
pc_save = o3d.geometry.PointCloud()
pc_save.points = o3d.utility.Vector3dVector(pc_filtered_nparray)
o3d.io.write_point_cloud(filtered_point_cloud_save_dir, pc_save)

cloud_deground, ground, ground_model = pcl_extract_ground(pc_filtered_pcl)
# visualize ground and deground point cloud
colors = np.zeros((cloud_deground.shape[0] + ground.shape[0], 3))
colors[:cloud_deground.shape[0], 0] = 1
colors[cloud_deground.shape[0]:, 1] = 1
pc_o3d.points = o3d.utility.Vector3dVector(np.append(cloud_deground, ground, axis=0))
pc_o3d.colors = o3d.utility.Vector3dVector(colors)
# o3d.visualization.draw_geometries([pc_o3d])

o3d.io.write_point_cloud(extract_ground_point_cloud_save_dir, pc_o3d)

# Euclidean clustering
# Creating the KdTree object for the search method of the extraction
cloud_deground_pcl = pcl.PointCloud()
cloud_deground_pcl.from_array(cloud_deground)
tree = cloud_deground_pcl.make_kdtree()
ec = cloud_deground_pcl.make_EuclideanClusterExtraction()
ec.set_ClusterTolerance (0.6)
ec.set_MinClusterSize (20)
ec.set_MaxClusterSize (10000)
ec.set_SearchMethod (tree)
cluster_indices = ec.Extract()

# draw bounding boxes
ec_pc_vis = np.zeros((0, 3))
ec_color_vis = np.zeros((0, 3))
bboxes_points_vis = np.zeros((0, 3))
cloud_cluster = pcl.PointCloud()
for j, indices in enumerate(cluster_indices):
    print('indices = ' + str(len(indices)))
    points = np.zeros((len(indices), 3), dtype=np.float32)

    for i, indice in enumerate(indices):
        points[i] = cloud_deground_pcl[indice]

    pc_center = (np.max(points, 0) + np.min(points, 0)) / 2
    if pc_center[2] < -1:
        print('pc_center is ', pc_center, ', do not consider.')
        continue


    colors = get_random_color_vec(len(indices))
    ec_pc_vis = np.append(ec_pc_vis, points, axis=0)
    ec_color_vis = np.append(ec_color_vis, colors, axis=0)
    bboxes_points_vis = np.append(bboxes_points_vis, draw_bbox_of_pc_cluster(points), axis=0)

# visualize point cloud
pc_vis = o3d.geometry.PointCloud()
pc_vis_array = np.zeros((0, 3))
color_vis_array = np.zeros((0, 3))

# add euclidean colored point cloud
pc_vis_array = np.append(pc_vis_array, ec_pc_vis, axis=0)
color_vis_array = np.append(color_vis_array, ec_color_vis, axis=0)
# add bboxes points to euclidean point cloud
pc_vis_array = np.append(pc_vis_array, bboxes_points_vis, axis=0)
color_vis_array = np.append(color_vis_array, [0,0,1] + np.zeros((bboxes_points_vis.shape[0], 3)), axis=0)
# add origin point cloud
# pc_vis_array = np.append(pc_vis_array, cloud_deground, axis=0)
# color_vis_array = np.append(color_vis_array, [0.9,0.9,0.9] + np.zeros((cloud_deground.shape[0], 3)), axis=0)


# draw_geometry
pc_vis.points = o3d.utility.Vector3dVector(pc_vis_array)
pc_vis.colors = o3d.utility.Vector3dVector(color_vis_array)
# o3d.visualization.draw_geometries([pc_vis])

o3d.io.write_point_cloud(extract_ground_clustered_point_cloud_save_dir, pc_vis)

