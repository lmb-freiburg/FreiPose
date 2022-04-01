from __future__ import print_function, unicode_literals
from collections import namedtuple
import os
import numpy as np
import cv2

from .general_util import preprocess_image, calc_bbox


def calib_to_list(calib, cam_range):
    M_list, K_list, dist_list = list(), list(), list()
    for cid in cam_range:
        cam_name = 'cam%d' % cid
        M_list.append(calib[cam_name]['M'])
        K_list.append(calib[cam_name]['K'])
        dist_list.append(calib[cam_name]['dist'])
    return K_list, dist_list, M_list


def anno_to_mat(anno_frame, cam_range, num_kp):
    num_cams = len(cam_range)
    kp_uv = np.zeros((num_cams, num_kp, 2))
    vis2d = np.zeros((num_cams, num_kp))

    for i, cid in enumerate(cam_range):
        if 'cam%d' % cid in anno_frame.keys():
            kp_uv[i] = np.array(anno_frame['cam%d' % cid]['kp_uv'])
            vis2d[i] = np.array(anno_frame['cam%d' % cid]['vis'])
    if 'xyz' in list(anno_frame.keys()):
        xyz = np.array(anno_frame['xyz'])
        vis3d = np.array(anno_frame['vis3d'])
    else:
        xyz=None
        vis3d=None
    return xyz, vis3d, kp_uv, vis2d


def check_if_labeled(anno, num_kp_thresh=1):
    """ Check if annotation actually contains usable information or is just empty. """
    for _, frame_data in anno.items():
        if is_good_2d(frame_data, num_kp_thresh):
            return True
    return False


def save_str(x):
    """ Convert string into a save one, ie removing problematic characters. """
    bad_chars = ['/', ':']
    for i in bad_chars:
        x = x.replace(i, '')
    return x


def get_ident(db):
    """ Calculate an identifier from a db entry. Kind of a hash value. """
    return '%s-%s-%s-%s' % (save_str(db['path']),
                            save_str(db['frame_dir']),
                            os.path.splitext(db['vid_file'])[0] if db['vid_file'] is not None else 'None',
                            os.path.splitext(db['anno'])[0])


def list_cams(base_path):
    items = os.listdir(base_path)
    items = [os.path.join(base_path, x) for x in items]
    items = [x for x in items if os.path.isdir(x) and 'cam' in x]
    return items, [os.path.basename(x) for x in items]


def list_frames(base_path):
    exts = ['.jpg', '.png', 'jpeg', '.bmp']  # probably not a complete list yet

    items = os.listdir(base_path)
    items = [x for x in items if any([ext in os.path.splitext(x)[1].lower() for ext in exts])]
    items = [os.path.join(base_path, x) for x in items]
    return items


def is_good_2d(anno_data, min_num_kp=10):
    if anno_data is not None:
        for k, v in anno_data.items():
            if 'cam' in k:
                if np.sum(v['vis']) >= min_num_kp:
                    return True
    return False


def is_good_3d(anno_data, min_num_kp=10):
    if anno_data is not None:
        # xyz = np.array(anno_data['xyz'])
        vis = np.array(anno_data['vis3d'])
        if np.sum(vis) >= min_num_kp:
            return True
    return False


def preproc_sample(base_path, cam_dirs, frame_name, points2d_merged, vis_merged, calib, crop_oversampling, crop_size):
    img_all = list()
    # xyz = np.array(anno_data['xyz'])
    scales_all, offsets_all = list(), list()
    for i, cam in enumerate(cam_dirs):
        img_crop, scale, offset = _preprocess(base_path, cam, frame_name, points2d_merged[i], vis_merged[i],
                                              calib[cam]['K'], calib[cam]['dist'], crop_oversampling, crop_size)
        scales_all.append(scale)
        offsets_all.append(offset)
        img_all.append(img_crop)

    return img_all, scales_all, offsets_all


def _preprocess(base_path, cam, frame_name, uv, vis, K, dist,
                crop_oversampling, crop_size):

    # calculate bounding box
    bbox = calc_bbox(uv, vis)

    img = cv2.imread(os.path.join(base_path, cam, frame_name))
    img = cv2.undistort(img, K, dist)
    img_crop, scale, offset, img_raw_crop = preprocess_image(img, bbox,
                                                             do_mean_subtraction=True,
                                                             over_sampling=crop_oversampling,
                                                             symmetric=True,
                                                             resize=True, target_size=crop_size,
                                                             raise_exp=False)

    img_raw_crop = img_raw_crop.astype(np.uint8)

    return img_raw_crop, scale, offset
