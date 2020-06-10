import rosbag
import IPython
import sensor_msgs.point_cloud2
import numpy as np
import os

bag_path = '/data/rangeimage_prediction_32E/pointcloud_rosbag/autoware_32e.bag'
save_dir = '/data/rangeimage_prediction_32E/pointcloud_txt_file/autoware_32e'

bag = rosbag.Bag(bag_path)

frame = 0
for topic, msg, t in bag.read_messages():
    # if frame <= 10:
    #     frame += 1
    #     continue
    if frame >= 30:
        break
    point_cloud = np.zeros((0, 3))
    if topic == 'velodyne_points':
        # print("\n\n\ntopic: ", topic)
        # print("\n\n\nt: ", t)
        # print("\n\n\nmsg: ", msg)
        num = 0
        for point in sensor_msgs.point_cloud2.read_points(msg, skip_nans=True):
            if num % 10000 == 0:
                print('compose %d points in frame %d.' %(num, frame))
            num += 1
            point_cloud = np.append(point_cloud, [[point[0], point[1], point[2]]], axis=0)
    save_path = os.path.join(save_dir, str(frame)+'.txt')
    np.savetxt(save_path, point_cloud, fmt='%.5f')
    frame += 1


bag.close()


