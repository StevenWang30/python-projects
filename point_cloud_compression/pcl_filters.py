import pcl
import IPython

# RadiusOutlierRemoval
def radius_outlier_removal_pcl(radius, min_pts, pointcloud):
    # not work
    pc_pcl = pcl.PointCloud(pointcloud)
    radius = 2
    filter = pc_pcl.make_RadiusOutlierRemoval()
    filter.set_radius_search(radius)
    # result = filter.get_radius_search()
    min_pts = 50
    filter.set_MinNeighborsInRadius(min_pts)
    # result = filter.get_MinNeighborsInRadius()

    result_point = filter.filter()

    # filter.set_negative(True)

    IPython.embed()

    # check
    # new instance is returned
    # assertNotEqual(p, result)
    # filter retains the same number of points
    # assertNotEqual(result_point.size, 0)
    # assertNotEqual(p.size, result_point.size)


### StatisticalOutlierRemovalFilter ###
def statistical_outlier_removal_filter(mean_k, pointcloud, negative=False):
    pc_pcl = pcl.PointCloud(pointcloud)
    filter = pc_pcl.make_statistical_outlier_filter()
    filter.set_mean_k(mean_k)
    filter.set_std_dev_mul_thresh(1.0)
    pos_points = filter.filter()
    filter.set_negative(True)
    neg_points = filter.filter()
    # IPython.embed()

    if negative:
        ret = neg_points
    else:
        ret = pos_points
    return ret


# ### VoxelGridFilter ###
# class TestVoxelGridFilter(unittest.TestCase):
#
#     def setUp(self):
#         p = pcl.load(
#             "tests" +
#             os.path.sep +
#             "table_scene_mug_stereo_textured_noplane.pcd")
#         fil = p.make_voxel_grid_filter()
#         # fil = pcl.VoxelGridFilter()
#         # fil.set_InputCloud(p)
#
#     def testFilter(self):
#         fil.set_leaf_size(0.01, 0.01, 0.01)
#         c = fil.filter()
#         assertTrue(c.size < p.size)
#         assertEqual(c.size, 719)
