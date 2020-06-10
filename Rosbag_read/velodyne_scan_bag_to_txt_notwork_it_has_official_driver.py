import rosbag
import IPython
import sensor_msgs.point_cloud2
import numpy as np

bag_path = '/data/rangeimage_prediction/DrivingData/Parking_lot.bag'
bag = rosbag.Bag(bag_path)

point_cloud = np.zeros((0,3))
for topic, msg, t in bag.read_messages():
    if topic == '/velodyne_packets':
        IPython.embed()
        # print("\n\n\ntopic: ", topic)
        # print("\n\n\nt: ", t)
        # print("\n\n\nmsg: ", msg)
        for point in sensor_msgs.point_cloud2.read_points(msg, skip_nans=True):
            point_cloud = np.append(point_cloud, [[point[0], point[1], point[2]]], axis=0)

    IPython.embed()
bag.close()