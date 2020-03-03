import numpy as np
import open3d as o3d
import os


path_list = [
	'/data/KITTI_object_tracking/training/velodyne/0019/000000.bin',
	'/data/KITTI_object_tracking/training/velodyne/0019/000004.bin',
	'/data/KITTI_object_tracking/training/velodyne/0019/000008.bin',
	'/data/KITTI_object_tracking/training/velodyne/0019/000012.bin',
	'/data/KITTI_object_tracking/training/velodyne/0019/000016.bin',
]

save_dir = '/data/KITTI_object_tracking/spatio-temporal-map/draw_pics/detection_3d'


def load_bin(file_name):
	scan = np.fromfile(file_name, dtype=np.float32)
	scan = scan.reshape((-1, 4))
	point_cloud_array = scan[:, :3]
	# pc = o3d.geometry.PointCloud()
	# pc.points = o3d.utility.Vector3dVector(point_cloud_array)
	return point_cloud_array


def draw_extraction_result(points, save_path, vis=False):
	points_o3d = o3d.geometry.PointCloud()
	points_o3d.points = o3d.utility.Vector3dVector(points)
	points_o3d.paint_uniform_color([1, 0, 0])
	if vis:
		o3d.visualization.draw_geometries([points_o3d])
	else:
		o3d.io.write_point_cloud(save_path, points_o3d)



for i in range(len(path_list)):
	bin_file = path_list[i]
	pc = load_bin(bin_file)
	save_file_name = 'Seq' + str(int(bin_file.split('/')[-2])) + '-' + bin_file.split('/')[-1].replace('.bin', '_velodyne.pcd')
	save_path = os.path.join(save_dir, save_file_name)
	print('save pcd file into ', save_path)
	draw_extraction_result(pc, save_path, vis=False)


