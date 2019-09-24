import pcl


# RadiusOutlierRemoval
def radius_outlier_removal_pcl(radius, pointcloud):
        pc_pcl = pcl.PointCloud(pointcloud)
        radius = radius
        filter = pc_pcl.make_RadiusOutlierRemoval()
        filter.set_radius_search(radius)
        result = filter.get_radius_search()

        min_pts = 5
        filter.set_MinNeighborsInRadius(min_pts)
        result = filter.get_MinNeighborsInRadius()

        result_point = filter.filter()

        # check
        # new instance is returned
        # self.assertNotEqual(self.p, result)
        # filter retains the same number of points
        # self.assertNotEqual(result_point.size, 0)
        # self.assertNotEqual(self.p.size, result_point.size)


### StatisticalOutlierRemovalFilter ###
class TestStatisticalOutlierRemovalFilter(unittest.TestCase):

    def setUp(self):
        self.p = pcl.load(
            "tests" +
            os.path.sep +
            "table_scene_mug_stereo_textured_noplane.pcd")
        self.fil = self.p.make_statistical_outlier_filter()
        # self.fil = pcl.StatisticalOutlierRemovalFilter()
        # self.fil.set_InputCloud(self.p)

    def _tpos(self, c):
        self.assertEqual(c.size, 22745)
        self.assertEqual(c.width, 22745)
        self.assertEqual(c.height, 1)
        self.assertTrue(c.is_dense)

    def _tneg(self, c):
        self.assertEqual(c.size, 1015)
        self.assertEqual(c.width, 1015)
        self.assertEqual(c.height, 1)
        self.assertTrue(c.is_dense)

    def testFilterPos(self):
        fil = self.p.make_statistical_outlier_filter()
        fil.set_mean_k(50)
        self.assertEqual(fil.mean_k, 50)
        fil.set_std_dev_mul_thresh(1.0)
        self.assertEqual(fil.stddev_mul_thresh, 1.0)
        c = fil.filter()
        self._tpos(c)

    def testFilterNeg(self):
        fil = self.p.make_statistical_outlier_filter()
        fil.set_mean_k(50)
        fil.set_std_dev_mul_thresh(1.0)
        self.assertEqual(fil.negative, False)
        fil.set_negative(True)
        self.assertEqual(fil.negative, True)
        c = fil.filter()
        self._tneg(c)

    def testFilterPosNeg(self):
        fil = self.p.make_statistical_outlier_filter()
        fil.set_mean_k(50)
        fil.set_std_dev_mul_thresh(1.0)
        c = fil.filter()
        self._tpos(c)
        fil.set_negative(True)
        c = fil.filter()
        self._tneg(c)


### VoxelGridFilter ###
class TestVoxelGridFilter(unittest.TestCase):

    def setUp(self):
        self.p = pcl.load(
            "tests" +
            os.path.sep +
            "table_scene_mug_stereo_textured_noplane.pcd")
        self.fil = self.p.make_voxel_grid_filter()
        # self.fil = pcl.VoxelGridFilter()
        # self.fil.set_InputCloud(self.p)

    def testFilter(self):
        self.fil.set_leaf_size(0.01, 0.01, 0.01)
        c = self.fil.filter()
        self.assertTrue(c.size < self.p.size)
        self.assertEqual(c.size, 719)
