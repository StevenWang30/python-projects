import numpy as np
from scipy.optimize import leastsq
import IPython

def least_square_x_y(points):
    '''
    quadratic curve equation in x and y direction:
    x = a*z^2 + b*z + c
    y = d*z^2 + e*z + f
    :param points: n * 3 [z, x, y] ([t,x,y])
    :return: [[a, b, c], [d, e, f]]
    '''
    ps = np.array(points)
    x = ps[:, 1]
    y = ps[:, 2]
    z = ps[:, 0]

    p0 = [10, 10, 10]

    Para = leastsq(error, p0, args=(z, x))
    a, b, c = Para[0]

    Para = leastsq(error, p0, args=(z, y))
    d, e, f = Para[0]

    solution = [[a, b, c], [d, e, f]]
    # print("solution: ", solution)
    return solution


# 误差函数，即拟合曲线所求的值与实际值的差
def error(params, x, y):
    a, b, c = params
    err = a * x * x + b * x + c - y
    return err


def test_least_square_x_y(points, L):
    ans = []
    for i in range(len(points)):
        x = points[i][1]
        y = points[i][2]
        z = points[i][0]
        ans.append([z, x, y, abs(error(L[0], z, x)), abs(error(L[1], z, y))])
    print(ans)


def evaluate_distance_x_y(data, L, threshold=2, frame='global'):
    min_dis = 999
    min_idx = -1
    if frame == 'global':
        location = data['global_location']
    else:
        location = data['location']
    for i in range(len(location)):
        x = location[i][0]
        y = location[i][1]
        z = data['metadata']['image_idx']
        dis = abs(error(L[0], z, x)) + abs(error(L[1], z, y))
        # print("location: ", data['global_location'][i], " distance: ", dis)
        if dis < min_dis and dis < threshold:
            min_dis = dis
            min_idx = i
    return min_idx, min_dis
