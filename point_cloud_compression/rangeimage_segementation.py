import numpy as np

def range_image_segementation(range_image):
    '''
    :param range_image: an nxm matrix with double depth of each pixel
          Example: 3x4 matrix   o - o - o - o   has (3-1)x4 vertical edges
                                |   |   |   |   and 3x(4-1) horizontal edges
                                o - o - o - o
                                |   |   |   |
                                o - o - o - o
        an nxm range_image has nxm nodes with (n-1)xm+(m-1)xn edges
        so we construct a (n-1)xm matrix and (m-1)xn matrix, stand for the vertical edges and horizontal edges.
        each edge's value is the depth differential of two adjacent pixels.
    :return: return a map, from left-top to right-bottom --> 0 to ..., different value means different cluster
    '''
    map = np.zeros_like(range_image)
    n = range_image.shape[0]
    m = range_image.shape[1]
    vertical_edges = np.zeros((n-1, m))
    horizontal_edges = np.zeros((m-1, n))
    vertical_edges = abs(range_image[:n-1, :] - range_image[1:, :])
    horizontal_edges = abs(range_image[:, m - 1] - range_image[:, 1:])

    # start_id = 1
    # for y in range(m):
    #     for x in range(n):
    #         if map[x, y] != 0:

