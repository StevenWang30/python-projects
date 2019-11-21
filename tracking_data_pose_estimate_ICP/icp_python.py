import numpy as np
import os
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
import IPython
import open3d as o3d
import copy

from sklearn.neighbors import NearestNeighbors

np.set_printoptions(suppress=True) #python 不以科学计数法输出

def draw_registration_result(source, target, transformation):
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    source_temp.paint_uniform_color([1, 0.706, 0])
    target_temp.paint_uniform_color([0, 0.651, 0.929])
    source_temp.transform(transformation)
    o3d.visualization.draw_geometries([source_temp, target_temp])

def icp_open3d(source, target, trans_init, threshold=1):
    # draw_registration_result(source, target, trans_init)
    # print("Initial alignment")
    # evaluation = o3d.registration.evaluate_registration(source, target, threshold, trans_init)
    # print(evaluation)

    # print("Apply point-to-point ICP")
    # reg_p2p = o3d.registration.registration_icp(
    #     source, target, threshold, trans_init,
    #     o3d.registration.TransformationEstimationPointToPoint())
    # print(reg_p2p)
    # # print("Transformation is:")
    # # print(reg_p2p.transformation)
    # # print("")
    # # draw_registration_result(source, target, reg_p2p.transformation)
    # transformed_pointcloud = copy.deepcopy(source)
    # transformed_pointcloud.transform(reg_p2p.transformation)

    print("Apply point-to-plane ICP")
    o3d.geometry.estimate_normals(source, search_param=o3d.geometry.KDTreeSearchParamHybrid(
        radius=0.3, max_nn=30))
    o3d.geometry.estimate_normals(target, search_param=o3d.geometry.KDTreeSearchParamHybrid(
        radius=0.3, max_nn=30))
    reg_p2l = o3d.registration.registration_icp(
        source, target, threshold, trans_init,
        o3d.registration.TransformationEstimationPointToPlane())
    print(reg_p2l)
    transformed_pointcloud = copy.deepcopy(source)
    transformed_pointcloud.transform(reg_p2l.transformation)
    #
    # print("Transformation is:")
    # print(reg_p2l.transformation)
    # print("")
    # # draw_registration_result(source, target, reg_p2l.transformation)

    return reg_p2l.transformation, transformed_pointcloud


def best_fit_transform(A, B):
    '''
    Calculates the least-squares best-fit transform that maps corresponding points A to B in m spatial dimensions
    Input:
      A: Nxm numpy array of corresponding points
      B: Nxm numpy array of corresponding points
    Returns:
      T: (m+1)x(m+1) homogeneous transformation matrix that maps A on to B
      R: mxm rotation matrix
      t: mx1 translation vector
    '''

    assert A.shape == B.shape

    # get number of dimensions
    m = A.shape[1]

    # translate points to their centroids
    centroid_A = np.mean(A, axis=0)
    centroid_B = np.mean(B, axis=0)
    AA = A - centroid_A
    BB = B - centroid_B

    # rotation matrix
    H = np.dot(AA.T, BB)
    U, S, Vt = np.linalg.svd(H)
    R = np.dot(Vt.T, U.T)

    # special reflection case
    if np.linalg.det(R) < 0:
       Vt[m-1,:] *= -1
       R = np.dot(Vt.T, U.T)

    # translation
    t = centroid_B.T - np.dot(R,centroid_A.T)

    # homogeneous transformation
    T = np.identity(m+1)
    T[:m, :m] = R
    T[:m, m] = t

    return T, R, t


def nearest_neighbor(src, dst):
    '''
    Find the nearest (Euclidean) neighbor in dst for each point in src
    Input:
        src: Nxm array of points
        dst: Nxm array of points
    Output:
        distances: Euclidean distances of the nearest neighbor
        indices: dst indices of the nearest neighbor
    '''

    assert src.shape == dst.shape

    neigh = NearestNeighbors(n_neighbors=1)
    neigh.fit(dst)
    distances, indices = neigh.kneighbors(src, return_distance=True)
    return distances.ravel(), indices.ravel()


def icp(A, B, init_pose=None, max_iterations=20, tolerance=0.001):
    '''
    The Iterative Closest Point method: finds best-fit transform that maps points A on to points B
    Input:
        A: Nxm numpy array of source mD points
        B: Nxm numpy array of destination mD point
        init_pose: (m+1)x(m+1) homogeneous transformation
        max_iterations: exit algorithm after max_iterations
        tolerance: convergence criteria
    Output:
        T: final homogeneous transformation that maps A on to B
        distances: Euclidean distances (errors) of the nearest neighbor
        i: number of iterations to converge
    '''
    # IPython.embed()
    if(A.shape != B.shape):
        min = np.min((A.shape[0], B.shape[0]))
        A = A[:min, :]
        B = B[:min, :]

    assert A.shape == B.shape

    # get number of dimensions
    m = A.shape[1]

    # make points homogeneous, copy them to maintain the originals
    src = np.ones((m+1,A.shape[0]))
    dst = np.ones((m+1,B.shape[0]))
    src[:m,:] = np.copy(A.T)
    dst[:m,:] = np.copy(B.T)

    # apply the initial pose estimation
    if init_pose is not None:
        src = np.dot(init_pose, src)

    prev_error = 0

    for i in range(max_iterations):
        # find the nearest neighbors between the current source and destination points
        distances, indices = nearest_neighbor(src[:m,:].T, dst[:m,:].T)

        # compute the transformation between the current source and nearest destination points
        T,_,_ = best_fit_transform(src[:m,:].T, dst[:m,indices].T)

        # update the current source
        src = np.dot(T, src)

        # check error
        mean_error = np.mean(distances)
        if np.abs(prev_error - mean_error) < tolerance:
            break
        prev_error = mean_error

    # calculate final transformation
    T,_,_ = best_fit_transform(A, src[:m,:].T)

    T = np.array(T)
    print(T)

    origin_matrix = np.zeros((4,4))
    origin_matrix[3][3] = 1.0236
    origin_matrix[0][3] = 0.7554
    origin_matrix[1][3] = 1.4264
    origin_matrix[2][3] = 1
    wx = 1.0236 - 1.0249
    wy = 0.7554 - 0.7561
    wz = 1.4264 - 1.4260
    x = 0.0536 - 0.0540
    y = 0.8675 - 0.8675
    z = -0.4929 - -0.4928
    w = -0.0407 - -0.0413
    om = np.asarray([[1-2*y*y-2*z*z, 2*x*y+2*w*z, 2*x*z-2*w*y, wx],
                     [2*x*y-2*w*z,  1-2*x*x-2*z*z,2*y*z+2*w*x, wy],
                     [2*x*z+2*w*y, 2*y*z+2*w*x, 1-2*x*x-2*y*y, wz],
                     [0,            0,          0,             1]])
    print(om)


    return T, distances, i

if __name__ == "__main__":
    print('pid: %s'%(str(os.getpid())))

    data_path = "/home/skwang/data/KITTI_rawdata/2011_09_26_drive_0009_extract/velodyne_points_ground_extractioin/point_cloud"
    pc1_name = "0000000300.pcd"
    pc2_name = "0000000301.pcd"
    source = o3d.io.read_point_cloud(os.path.join(data_path, pc1_name))
    target = o3d.io.read_point_cloud(os.path.join(data_path, pc2_name))
    trans_init = np.asarray([[1.0, 0.0, 0.0, 0.0],
                             [0.0, 1.0, 0.0, 0.0],
                             [0.0, 0.0, 1.0, 0.0],
                             [0.0, 0.0, 0.0, 1.0]])
    icp_open3d(source, target, trans_init)

    # A_path = "/data/TUM_dataset/rgbd_dataset_freiburg2_desk_with_person/test/xyz1311870537.974647.txt"
    # B_path = "/data/TUM_dataset/rgbd_dataset_freiburg2_desk_with_person/test/xyz1311870538.002578.txt"
    # A = np.genfromtxt(A_path)
    # B = np.genfromtxt(B_path)
    # A = A[:, 0:3]
    # B = B[:, 0:3]
    # icp(A, B, max_iterations=2000, tolerance=0.000001)
