import argparse, os
import numpy as np
import cv2, time

from config.Model import Model
from utils.general_util import json_load, my_mkdir

from predict_bb import parse_input


def read_vid_frame(video_path, fid):
    """ Reads a single frame from a video.
    """
    cap = cv2.VideoCapture(video_path)
    vid_size = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    assert 0 <= fid < vid_size, 'Frame id is outside the video.'

    cap.set(cv2.CAP_PROP_POS_FRAMES, fid)
    for i in range(5):
        suc, img = cap.read()
        if not suc:
            print('Reading video frame was not successfull. Will try again in 2 sec.')
            time.sleep(2)
        else:
            break
    assert img is not None and suc, 'Reading not successful'
    cap.release()
    return img


def _load(pred_file, label_file, mask=True):
    pred_data = json_load(pred_file)
    print('Loaded predictions for %d frames from %s' % (len(pred_file), pred_file))

    label_data = json_load(label_file)
    print('Loaded labels for %d frames from %s' % (len(label_data), label_file))

    # find common subset
    common = list()
    for k, v in label_data.items():
        i = int(os.path.splitext(k)[0])
        common.append(
            [k, i]
        )
    print('Found %d frames in common' % len(common))

    pred_xyz, gt_xyz, gt_vis = list(), list(), list()
    for k, i in common:
        gt_vis.append(label_data[k]['vis3d'])
        gt_xyz.append(label_data[k]['xyz'])
        pred_xyz.append(pred_data[i]['kp_xyz'])
    pred_xyz, gt_xyz, gt_vis = np.array(pred_xyz), np.array(gt_xyz), np.array(gt_vis)
    pred_xyz = np.reshape(pred_xyz, gt_xyz.shape)

    # mask to only valid ones
    if mask:
        m = gt_vis > 0.5
        pred_xyz_m = pred_xyz[m]
        gt_xyz_m = gt_xyz[m]

    return pred_xyz_m, gt_xyz_m, common, pred_xyz, gt_xyz


def _dump_vis(model, pred, gt, common,
              video_list, K_list, dist_list, M_list):
    from utils.plot_util import draw_skel
    from utils.StitchedImage import StitchedImage
    import utils.CamLib as cl
    from tqdm import tqdm

    # iterate frames
    for i, (_, fid) in tqdm(enumerate(common), desc='Dumping Samples', total=len(common)):
        # Accumulate frames
        merged_list = list()
        # inpaint pred/gt
        for K, dist, M, v in zip(K_list, dist_list, M_list, video_list):
            img = read_vid_frame(v, fid)
            uv_p = cl.project(cl.trafo_coords(pred[i], M), K, dist)
            img_p = draw_skel(img.copy(), model, uv_p, color_fixed='r', order='uv')
            uv_gt = cl.project(cl.trafo_coords(gt[i], M), K, dist)
            img_p = draw_skel(img_p, model, uv_gt, color_fixed='g', order='uv')

            merged_list.append(img_p)

        merged = StitchedImage(merged_list)
        p = os.path.join(os.path.dirname(video_list[0]), 'eval_vis_dump/%04d.png' % i)
        # cv2.imshow('img', merged.image)
        # cv2.waitKey()
        my_mkdir(p, is_file=True)
        cv2.imwrite(p, merged.image)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Calculate evaluation error of network predictions wrt labels.')
    parser.add_argument('model', type=str, help='Model definition file.')
    parser.add_argument('pred_file', type=str, help='Prediction file of sequence.')
    parser.add_argument('label_file', type=str, help='Label file containing annotations.')
    parser.add_argument('--video_path', type=str, help='If given creates some visualization of the predictions.')

    parser.add_argument('--run_wildcard', type=str, default='run%03d', help='How to tell the run id'
                                                                            ' from a given file name.')
    parser.add_argument('--cam_wildcard', type=str, default='cam%d', help='How to tell the camera id'
                                                                          ' from a given file name.')
    parser.add_argument('--max_cams', type=int, default=64, help='Maximal number of cams we search for.')
    parser.add_argument('--calib_file_name', type=str, default='M.json', help='Assumed calibration file name.')
    args = parser.parse_args()

    # load model data
    model = Model(args.model)

    # load inputs
    pred, gt,\
    common, pred_raw, gt_raw = _load(args.pred_file, args.label_file)

    # calculate error
    error = np.linalg.norm(pred - gt, 2, -1)
    error = error.mean()*1000.0
    print('Resulting MPJPE %.2f mm' % error)

    if args.video_path is not None:
        # parse given input
        video_list, K_list, \
        dist_list, M_list, _ = parse_input(args.video_path,
                                           args.cam_wildcard, args.run_wildcard, args.max_cams,
                                           args.calib_file_name,
                                           None)

        # render examples
        _dump_vis(model, pred_raw, gt_raw, common,
                  video_list, K_list, dist_list, M_list)

