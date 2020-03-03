import numpy as np
# from sklearn.utils.linear_assignment_ import linear_assignment
from scipy.optimize import linear_sum_assignment
import IPython


def sort(cost_matrix, threshold=7.0):
    """
    Returns 3 lists of matches, unmatched_detections and unmatched_trackers
    """

    cost_matrix[cost_matrix > 6] = 100
    # matched_indices_old = linear_assignment(cost_matrix)
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    matched_indices = np.append(np.expand_dims(row_ind, axis=-1), np.expand_dims(col_ind, axis=-1), axis=-1)
    # if not np.array_equal(matched_indices_old, matched_indices):
    #     print(matched_indices_old)
    #     print(matched_indices)
    #     print(np.array_equal(matched_indices_old, matched_indices))
    #     IPython.embed()

    # if cost_matrix.shape[0] - len(matched_indices) > 3:
    #     IPython.embed()

    unmatched_frame_1 = []
    for d in range(cost_matrix.shape[0]):
        if d not in matched_indices[:, 0]:
            unmatched_frame_1.append(d)
    unmatched_frame_2 = []
    for t in range(cost_matrix.shape[1]):
        if t not in matched_indices[:, 1]:
            unmatched_frame_2.append(t)

    # filter out matched with low IOU
    matches = []
    for m in matched_indices:
        if cost_matrix[m[0], m[1]] > threshold:
            unmatched_frame_1.append(m[0])
            unmatched_frame_2.append(m[1])
        else:
            matches.append(m.reshape(1, 2))
    if len(matches) == 0:
        matches = np.empty((0, 2), dtype=int)
    else:
        matches = np.concatenate(matches, axis=0)

    print('matches_pairs:', matches)
    print('unmatch object in frame_1:', unmatched_frame_1)
    print('unmatch object in frame_2:', unmatched_frame_2)

    return matches, np.array(unmatched_frame_1), np.array(unmatched_frame_2)
