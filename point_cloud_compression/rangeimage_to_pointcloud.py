import numpy as np
import IPython


def rangeimage_to_pointcloud(rangeimage, lidar_angular_xy_range, max_lidar_angular_z, min_lidar_angular_z, nearest_bound, furthest_bound):
    range_x = rangeimage.shape[0]
    range_y = rangeimage.shape[1]
    resolution_z = (max_lidar_angular_z - min_lidar_angular_z) / range_x / 180 * np.pi
    resolution_xy = (lidar_angular_xy_range / 180 * np.pi) / range_y

    rangeimage[np.where(rangeimage < nearest_bound)] = 0
    rangeimage[np.where(rangeimage > furthest_bound)] = 0

    # z_angle = max_lidar_angular_z / 180 * np.pi - x * resolution_z
    z_angle = np.arange(range_x).reshape(range_x, 1) + np.zeros((range_x, range_y))
    z_angle = max_lidar_angular_z / 180 * np.pi - z_angle * resolution_z
    xy_angle = np.arange(range_y).reshape(1, range_y) + np.zeros((range_x, range_y))
    xy_angle = xy_angle * resolution_xy
    world_x = (rangeimage * np.sin(z_angle)).reshape((range_x * range_y, 1))
    world_y = (rangeimage * np.cos(z_angle) * np.cos(xy_angle)).reshape((range_x * range_y, 1))
    world_z = (rangeimage * np.cos(z_angle) * np.sin(xy_angle)).reshape((range_x * range_y, 1))
    pointcloud = np.concatenate((world_x, world_y), axis=1)
    pointcloud = np.concatenate((pointcloud, world_z), axis=1)
    return pointcloud


def rangeimage_to_pointcloud_old_version(rangeimage, lidar_angular_xy_range, max_lidar_angular_z, min_lidar_angular_z, nearest_bound, furthest_bound):
    range_x = rangeimage.shape[0]
    range_y = rangeimage.shape[1]
    resolution_z = (max_lidar_angular_z - min_lidar_angular_z) / range_x / 180 * np.pi
    resolution_xy = (lidar_angular_xy_range / 180 * np.pi) / range_y
    pointcloud = np.zeros((0,3))
    for x in range(range_x):
        for y in range(range_y):
            depth = rangeimage[x, y]
            if depth < nearest_bound or depth > furthest_bound:
                print("point ",x,y," is out of the range of the 3D-lidar's range.")
                continue
            z_angle = max_lidar_angular_z / 180 * np.pi - x * resolution_z
            xy_angle = resolution_xy * y
            world_z = depth * np.sin(z_angle)
            world_x = depth * np.cos(z_angle) * np.cos(xy_angle)
            world_y = depth * np.cos(z_angle) * np.sin(xy_angle)
            pointcloud = np.append(pointcloud, [[world_x, world_y, world_z]], axis=0)
    return pointcloud