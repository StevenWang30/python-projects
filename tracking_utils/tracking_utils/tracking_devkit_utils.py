import os
from KITTI_dataset_utils.utils import *
from shutil import copyfile
from tracking_utils.dataset import *
from tracking_utils.visualization_utils import *
# import glob


def resume_all_original_label_file(root):
    label_dir = os.path.join(root, 'label_02')
    labels = os.listdir(label_dir)
    dest_dir = '/data/KITTI_object_tracking/devkit/python/data/tracking/label_02/'
    for label_i in labels:
        copyfile(os.path.join(label_dir, label_i), os.path.join(dest_dir, label_i))


def get_devkit_result(exp_result_save_dir, pred_save_root, method_name, sequence_start, sequence_end, Changed_Variable=0):
    temp_dir = os.getcwd()
    src = '/data/KITTI_object_tracking/devkit/python'
    os.chdir(src)

    for i in range(sequence_start, sequence_end+1):
        seq_str = str(i).zfill(4) + '.txt'
        pred_dir = os.path.join(pred_save_root, method_name, seq_str)
        dest_dir = os.path.join(src, 'results', 'pred', 'data', seq_str)
        copyfile(pred_dir, dest_dir)

    output = os.popen('python2 evaluate_tracking.py pred').read()
    with open(exp_result_save_dir, 'a+') as f:
        f.write('\n\n\n-------------------------- Method: ' + method_name + ' -------------------------- \n')
        f.write('                     Changed Variable = ' + str(Changed_Variable) + '  \n')
        # f.write(output)
        output_split = output.split('\n')
        for line in range(len(output_split)):
            line_content = output_split[line]
            if 'Multiple Object Tracking Accuracy (MOTA)' in line_content:
                f.write(line_content + '\n')
            elif 'Multiple Object Tracking Precision (MOTP)' in line_content:
                f.write(line_content + '\n')
            elif 'Mostly Tracked' in line_content:
                f.write(line_content + '\n')
            elif 'Mostly Lost' in line_content:
                f.write(line_content + '\n')
            elif 'ID-switches' in line_content:
                f.write(line_content + '\n')
            elif 'Fragmentations' in line_content:
                f.write(line_content + '\n')
    f.close()
    print('Finish writing into ', exp_result_save_dir)

    os.chdir(temp_dir)


def get_devkit_result_compare(exp_result_save_dir, pred_name, seq, Changed_Variable=0):
    temp_dir = os.getcwd()
    src = '/data/KITTI_object_tracking/devkit/python'
    os.chdir(src)

    seq_str = str(seq).zfill(4) + '.txt'
    pred_dir = os.path.join(save_path_root, 'pred', 'data', pred_name)
    dest_dir = os.path.join(src, 'results', 'pred', 'data', seq_str)
    copyfile(pred_dir, dest_dir)

    output = os.popen('python2 evaluate_tracking.py pred').read()
    with open(exp_result_save_dir, 'a+') as f:
        f.write('\n\n\n-------------------------- ' + pred_name.replace('.txt', '') + ' -------------------------- \n')
        f.write('                     Changed Variable = ' + str(Changed_Variable) + '  \n')
        output_split = output.split('\n')
        for line in range(len(output_split)):
            line_content = output_split[line]
            if 'Multiple Object Tracking Accuracy (MOTA)' in line_content:
                f.write(line_content + '\n')
            elif 'Multiple Object Tracking Precision (MOTP)' in line_content:
                f.write(line_content + '\n')
            elif 'Mostly Tracked' in line_content:
                f.write(line_content + '\n')
            elif 'Mostly Lost' in line_content:
                f.write(line_content + '\n')
            elif 'ID-switches' in line_content:
                f.write(line_content + '\n')
            elif 'Fragmentations' in line_content:
                f.write(line_content + '\n')
    f.close()
    print('Using tracking result:')
    print(output)

    os.chdir(temp_dir)


def save_add_displacement_label(save_path, sequence, frame, displacement, T_cam_velo):
    origin_label_root = '/data/KITTI_object_tracking/training/label_02'
    sequence = '%04d' % sequence
    origin_label_file = os.path.join(origin_label_root, sequence + '.txt')
    lines = [line.rstrip() for line in open(origin_label_file)]
    size = np.zeros((0,3))
    center = np.zeros((0,3))
    heading_angle = np.zeros(0)
    object_id = np.zeros(0)
    alpha_label = np.zeros(0)
    for l in range(len(lines)):
        frame_i = int(lines[l].split(' ')[0])
        if frame_i < frame:
            continue
        if frame_i > frame:
            break
        label_i = read_line(lines[l])
        if label_i['type'] in type_whitelist:
            xyz = np.array(label_i['t'])
            xyz_center_cam = xyz - [0, label_i['h'] / 2, 0]
            xyz_center = np.dot(T_cam_velo, np.append(xyz_center_cam, 1))[:3]
            # lwh = np.array([label_i['l'], label_i['w'], label_i['h']])

            size = np.append(size, [[label_i['l'], label_i['w'], label_i['h']]], axis=0)
            center = np.append(center, [xyz_center], axis=0)
            heading_angle = np.append(heading_angle, label_i['ry'])
            object_id = np.append(object_id, label_i['track_id'])
            alpha_label = np.append(alpha_label, label_i['alpha'])
    # IPython.embed()
    center += displacement

    center_camera = transform_points(center, np.linalg.inv(T_cam_velo))
    # center_camera = center
    calib_root = '/data/KITTI_object_tracking/training/calib'
    calibfile = os.path.join(calib_root, sequence + '.txt')
    data, _ = get_calibration(calibfile)
    P = data['P2'].reshape((3, 4))
    file = open(save_path, 'a+')
    # file = open(os.path.join(result_save_path, str(EVAL_STEP_ * BATCH_SIZE + i).zfill(6)+'.txt'), 'a')
    for j in range(size.shape[0]):
        if not np.any(size):
            break
        track_id = object_id[j]

        type = 'Car'
        trash1 = 0
        trash2 = 0
        beta = np.arctan(center_camera[j][0] / center_camera[j][2])
        # alpha = ((-((np.pi + heading_angle[j]) + (np.pi + beta))) % (2 * np.pi)) - np.pi
        alpha = alpha_label[j]
        bbox_points = get_bboxes_points(size[j], heading_angle[j], center[j], T_cam_velo)
        bbox_points_cam = transform_points(bbox_points, np.linalg.inv(T_cam_velo))
        bbox_points_cam = np.append(bbox_points_cam, np.ones((8, 1)), axis=1)
        bbox_2d = np.transpose(np.dot(P, np.transpose(bbox_points_cam)))
        bbox_2d[:, 0] = bbox_2d[:, 0] / bbox_2d[:, 2]
        bbox_2d[:, 1] = bbox_2d[:, 1] / bbox_2d[:, 2]
        bbox_2d = bbox_2d[:, :2]
        # left = (bbox_2d[0, 0] + bbox_2d[1, 0]) / 2
        # top = (bbox_2d[0, 1] + bbox_2d[1, 1]) / 2
        # right = (bbox_2d[2, 0] + bbox_2d[3, 0]) / 2
        # bottom = (bbox_2d[4, 1] + bbox_2d[5, 1]) / 2
        # IPython.embed()
        left = np.min(bbox_2d[:, 0])
        if left < 0:
            # print("wrong left :", left)
            left = 0
            # continue
        bottom = np.max(bbox_2d[:, 1])
        if bottom > 370:
            # print("wrong bottom :", bottom)
            bottom = 370
            # continue
        right = np.max(bbox_2d[:, 0])
        if right > 1224:
            # print("wrong right :", right)
            right = 1224
            # continue
        top = np.min(bbox_2d[:, 1])
        if top < 0:
            # print("wrong top :", top)
            top = 0
            # continue
        dimensions = size[j]
        if dimensions[0] < 2:
            print("dimension 0 < 2: ", dimensions[0])
            continue
        elif dimensions[0] < 3:
            print("dimension 0 < 3 and > 2: ", dimensions[0])
            dimensions[0] = 3.3
        if np.any(dimensions < 0.01):
            print("have dimension < 0.01: ", dimensions)
            continue

        # print("final image bbox:", [left, top, right, bottom])

        location = center_camera[j] + [0, dimensions[2] / 2, 0]
        rotation_y = heading_angle[j]

        file.write(str(int(frame)))
        file.write(' ')
        file.write(str(int(track_id)))
        file.write(' ')
        file.write(type)
        file.write(' ')
        file.write(str(round(trash1, 6)))
        file.write(' ')
        file.write(str(round(trash2, 6)))
        file.write(' ')
        file.write(str(round(alpha, 6)))
        file.write(' ')
        file.write(str(round(left, 6)))
        file.write(' ')
        file.write(str(round(top, 6)))
        file.write(' ')
        file.write(str(round(right, 6)))
        file.write(' ')
        file.write(str(round(bottom, 6)))
        file.write(' ')
        file.write(str(round(dimensions[2], 6)))
        file.write(' ')
        file.write(str(round(dimensions[1], 6)))
        file.write(' ')
        file.write(str(round(dimensions[0], 6)))
        assert np.all(dimensions != 0)
        file.write(' ')
        file.write(str(round(location[0], 6)))
        file.write(' ')
        file.write(str(round(location[1], 6)))
        file.write(' ')
        file.write(str(round(location[2], 6)))
        file.write(' ')
        file.write(str(round(rotation_y, 6)))
        file.write('\n')
    file.close()

def change_evaluate_tracking_seqmap_file(sequence_start, sequence_end):
    sequence_data = ['0000 empty 000000 000154',
                     '0001 empty 000000 000447',
                     '0002 empty 000000 000233',
                     '0003 empty 000000 000144',
                     '0004 empty 000000 000314',
                     '0005 empty 000000 000297',
                     '0006 empty 000000 000270',
                     '0007 empty 000000 000800',
                     '0008 empty 000000 000390',
                     '0009 empty 000000 000803',
                     '0010 empty 000000 000294',
                     '0011 empty 000000 000373',
                     '0012 empty 000000 000078',
                     '0013 empty 000000 000340',
                     '0014 empty 000000 000106',
                     '0015 empty 000000 000376',
                     '0016 empty 000000 000209',
                     '0017 empty 000000 000145',
                     '0018 empty 000000 000339',
                     '0019 empty 000000 001059',
                     '0020 empty 000000 000837']
    evaluate_tracking_seqmap_file_path = '/data/KITTI_object_tracking/devkit/python/data/tracking/evaluate_tracking.seqmap'
    file_root = evaluate_tracking_seqmap_file_path.strip(evaluate_tracking_seqmap_file_path.split('/')[-1])
    if os.path.exists(file_root):
        with open(evaluate_tracking_seqmap_file_path, 'w') as f:
            for i in range(sequence_start, sequence_end+1):
                f.write(sequence_data[i])
                f.write('\n')
        f.close()
