import numpy as np
import IPython

def least_square(points):
    '''
    quadratic curve equation:
    a*x^2 + b*x*y + c*x*z + d*y^2 + e*y*z + f*z^2 + g = 0
    :param points: n * 3 [z, x, y] ([t,x,y])
    :return: [a, b, c, d, e, f, g]
    '''
    A = []  # matrix
    for i in range(len(points)):
        x = points[i][1]
        y = points[i][2]
        z = points[i][0]
        A.append([x*x, x*y, x*z, y*y, y*z, z*z, 1])

    # solve this equation using svd
    s, v, d = np.linalg.svd(A)
    solution_1d = d[-1]

    # FIXME: not correct, test: the g always be larger than 0.998, so the x,y,z is real not important.
    IPython.embed()
    return solution_1d


def test_least_square(points, L):
    ans = []
    for i in range(len(points)):
        x = points[i][1]
        y = points[i][2]
        z = points[i][0]
        ans.append([z, x, y, abs(L[0]*x*x + L[1]*x*y + L[2]*x*z + L[3]*y*y + L[4]*y*z + L[5]*z*z + L[6]*1)])
    print(ans)


def evaluate_distance(data, L, threshold=2):
    min_dis = 999
    min_idx = -1
    for i in range(len(data['global_location'])):
        x = data['global_location'][i][0]
        y = data['global_location'][i][1]
        z = data['metadata']['image_idx']
        dis = abs(L[0]*x*x + L[1]*x*y + L[2]*x*z + L[3]*y*y + L[4]*y*z + L[5]*z*z + L[6]*1)
        # print("location: ", data['global_location'][i], " distance: ", dis)
        if dis < min_dis and dis < threshold:
            min_dis = dis
            min_idx = i
    return min_idx, min_dis
