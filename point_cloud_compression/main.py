
import open3d as o3d
from pointcloud_to_rangeimage import *
import pcl
import IPython
from pcl_extract_ground import pcl_extract_ground
from utils.utils import *
from draw_bbox import draw_bbox_of_pc_cluster
from PIL import Image
# import scipy.misc
import imageio

data_dir = './example_data/raw_point_cloud.pcd'
clustered_point_cloud_save_dir = './results/ec_point_cloud.pcd'
range_image_raw_save_dir = './results/range_image_raw.png'

pc_o3d = o3d.io.read_point_cloud(data_dir)
# o3d.visualization.draw_geometries([pc_o3d])
pc_nparray = np.asarray(pc_o3d.points).astype(np.float32)

Compose_Point_Cloud = False
if Compose_Point_Cloud:
    # Extract ground
    pc_pcl = pcl.PointCloud(pc_nparray)
    cloud_filtered, ground, ground_model = pcl_extract_ground(pc_pcl)
    # visualize ground and deground point cloud
    colors = np.zeros((cloud_filtered.shape[0] + ground.shape[0], 3))
    colors[:cloud_filtered.shape[0], 0] = 1
    colors[cloud_filtered.shape[0]:, 1] = 1
    pc_o3d.points = o3d.utility.Vector3dVector(np.append(cloud_filtered, ground, axis=0))
    pc_o3d.colors = o3d.utility.Vector3dVector(colors)
    # o3d.visualization.draw_geometries([pc_o3d])

    # Euclidean clustering
    # Creating the KdTree object for the search method of the extraction
    cloud_filtered_pcl = pcl.PointCloud()
    cloud_filtered_pcl.from_array(cloud_filtered)
    tree = cloud_filtered_pcl.make_kdtree()
    ec = cloud_filtered_pcl.make_EuclideanClusterExtraction()
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
            points[i] = cloud_filtered_pcl[indice]

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
    pc_vis_array = np.append(pc_vis_array, pc_nparray, axis=0)
    color_vis_array = np.append(color_vis_array, [0.9,0.9,0.9] + np.zeros((pc_nparray.shape[0], 3)), axis=0)


    # draw_geometry
    pc_vis.points = o3d.utility.Vector3dVector(pc_vis_array)
    pc_vis.colors = o3d.utility.Vector3dVector(color_vis_array)
    o3d.visualization.draw_geometries([pc_vis])

    o3d.io.write_point_cloud(clustered_point_cloud_save_dir, pc_vis)


    # cloud_cluster.from_array(points)
    # ss = "cloud_cluster_" + str(j) + ".pcd"
    # pcl.save(cloud_cluster, ss)
# IPython.embed()

Compose_Raw_Range_Image = True
if Compose_Raw_Range_Image:
    lidar_angular_xy_range_ = 360
    max_lidar_angular_z_ = 2
    min_lidar_angular_z_ = -24.5
    range_x_ = 64
    range_y_ = 2000
    nearest_bound_ = 0.5
    furthest_bound_ = 120
    if_show_ground_ = True
    range_image_array = pointcloud_to_rangeimage(pc_nparray, lidar_angular_xy_range_, max_lidar_angular_z_, min_lidar_angular_z_, range_x_, range_y_)
    range_image_array = remove_balck_line_and_remote_points(range_image_array)
    IPython.embed()
    # im = Image.fromarray(range_image_array).convert('RGB')
    # im.save(range_image_save_dir)
    imageio.imwrite(range_image_raw_save_dir, range_image_array)
