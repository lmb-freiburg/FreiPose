import argparse, os
import numpy as np
import cv2
from scipy.signal import convolve

from utils.general_util import json_dump, json_load
from utils.VideoReaderFast import VideoReaderFast
from utils.VideoWriterFast import *

from predict_bb import preprocess
from utils.general_util import load_calib_data, try_to_match

import utils.CamLib as cl


def load(video_file, cam_template):
    # figure out cam id
    video_name = os.path.basename(video_file)
    given_cid = try_to_match(video_name, cam_template)
    assert given_cid is not None, 'Cam id could not be extracted.'

    # find available cams
    record_name = None
    video_list, cam_range = list(), list()
    for cid in range(36):
        if record_name is None:
            record_name = os.path.splitext(video_name)[0]
            record_name = record_name.replace(cam_template % given_cid, '')

        test_path = video_file.replace(
            cam_template % given_cid,
            cam_template % cid
        )
        if os.path.exists(test_path):
            video_list.append(test_path)
            cam_range.append(cid)

    # try to find calib
    calib_file_name = os.path.join(
        os.path.dirname(video_file), 'M.json'
    )
    assert os.path.exists(calib_file_name), 'Calibration file not found.'

    # load and check calibration
    calib = load_calib_data(calib_file_name, return_cam2world=False)
    assert all(['cam%d' % cid in calib.keys() for cid in cam_range]), 'Missing calibration data for at least one camera.'

    # turn into lists
    K_list = [np.array(calib['cam%d' % i]['K']) for i in cam_range]
    dist_list = [np.array(calib['cam%d' % i]['dist']) for i in cam_range]
    M_list = [np.array(calib['cam%d' % i]['M']) for i in cam_range]

    return video_list, K_list, dist_list, M_list


def _get(pred, i):
    if pred[i] is None:
        # return average if not available
        return 0.5*(_get(pred, i-1) + _get(pred, i+1))

    if i < 0:
        return np.reshape(np.array(pred[i], [-1, 3]))
    if i >= len(pred):
        return np.reshape(np.array(pred[-1], [-1, 3]))
    else:
        return np.reshape(np.array(pred[i]), [-1, 3])


def smooth(pred):
    # make complete
    pred_complete = np.stack([_get(pred, i) for i in range(len(pred))])

    # smooth
    N = 3
    pred_complete_sm = convolve(pred_complete, np.ones((N, 1, 1))/3.0, mode='same')

    # import matplotlib.pyplot as plt
    # fig, axes = plt.subplots(3, 1)
    # axes[0].plot(pred_complete[:, 2, 0])
    # axes[0].plot(pred_complete_sm[:, 2, 0])
    # axes[1].plot(pred_complete[:, 2, 0])
    # axes[2].plot(pred_complete_sm[:, 2, 0])
    # plt.show()
    return pred_complete_sm


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Create small videos.')
    # parser.add_argument('output_path', type=str, help='Where the traj snippets are stored.')
    # parser.add_argument('video_file', type=str, help='Video to be used.')
    # parser.add_argument('pred_file', type=str, help='Pose predictions of out approach.')
    # parser.add_argument('pred_dlc_file', type=str, help='Pose predictions of DLC.')
    # parser.add_argument('traj_file', type=str, help='Traj definition file.')

    parser.add_argument('--cam_wildcard', type=str, default='cam%d', help='How to tell the camera id'
                                                                          ' from a given file name.')
    parser.add_argument('--kp_id', type=int, default=7)  #2 or 7
    parser.add_argument('--time', type=int, default=30)  # how many frame before and after are added to the video
    parser.add_argument('--top_n', type=int, default=5)  # how many frame before and after are added to the video
    args = parser.parse_args()

    # debug values
    args.video_file = '/misc/lmbraid18/datasets/RatPose/RatTrack_paper_resub_sessions/Rat506_200306/run046_cam1.avi'
    args.pred_file = '/misc/lmbraid18/datasets/RatPose/RatTrack_paper_resub_sessions/Rat506_200306/pred_run046__00.json'
    # args.pred_file = '/misc/lmbraid18/datasets/RatPose/RatTrack_paper_resub_sessions/Rat506_200306/artur_pred_run046__00.json'
    args.pred_dlc_file = '/misc/lmbraid18/datasets/RatPose/RatTrack_paper_resub_sessions/Rat506_200306/pred_dlc_unlabeled_Rat506_200306.json'
    # args.traj_file = '/misc/lmbraid18/datasets/RatPose/RatTrack_paper_resub_sessions/Rat506_200306/trigg_Rat506_200306_2.json'

    # args.video_file = '/misc/lmbraid18/datasets/RatPose/RatTrack_paper_resub_sessions/G372_190325/run001_cam1.mp4'
    # args.pred_file = '/misc/lmbraid18/datasets/RatPose/RatTrack_paper_resub_sessions/G372_190325/pred_run001__00.json'
    # args.pred_dlc_file = '/misc/lmbraid18/datasets/RatPose/RatTrack_paper_resub_sessions/G372_190325/pred_dlc_unlabeled_G372_190325.json'

    # args.video_file = '/misc/lmbraid18/datasets/RatPose/RatTrack_paper_resub_sessions/Rat487_200308/run060_cam1.avi'
    # args.pred_file = '/misc/lmbraid18/datasets/RatPose/RatTrack_paper_resub_sessions/Rat487_200308/pred_run060__00.json'
    # args.pred_dlc_file = '/misc/lmbraid18/datasets/RatPose/RatTrack_paper_resub_sessions/Rat487_200308/pred_dlc_unlabeled_Rat487_200308.json'

    # args.video_file = '/misc/lmbraid18/datasets/RatPose/RatTrack_paper_resub_sessions/Rat480_190823/run005_cam1.avi'
    # args.pred_file = '/misc/lmbraid18/datasets/RatPose/RatTrack_paper_resub_sessions/Rat480_190823/Rat480_190823_run005_pred_pose_ours_al1.json'
    # args.pred_dlc_file = '/misc/lmbraid18/datasets/RatPose/RatTrack_paper_resub_sessions/Rat480_190823/Rat480_190823_run005_pred_pose_dlc_al1.json'

    # load poses
    d = json_load(args.pred_file)
    pose_ours = list()
    for x in d:
        pose_ours.append(x.get('kp_xyz', None))

    # ours_raw = json_load(args.pred_file)
    # pose_ours = list()
    # k = list(ours_raw.keys())[0].split(':')[0]  # this will not win the price for the most readable code line
    # for i in range(len(ours_raw)):
    #     pose_ours.append(ours_raw['%s:%d' % (k, i)].get('kp_xyz', None))

    dlc_raw = json_load(args.pred_dlc_file)
    pose_dlc = list()
    k = list(dlc_raw.keys())[0].split(':')[0]  # this will not win the price for the most readable code line
    for i in range(len(dlc_raw)):
        pose_dlc.append(dlc_raw['%s:%d' % (k, i)].get('kp_xyz', None))
    print('pose_ours', len(pose_ours))
    print('pose_dlc', len(pose_dlc))
    pose_ours = smooth(pose_ours)
    pose_dlc = smooth(pose_dlc)

    # load trajectories
    # traj = json_load(args.traj_file)
    # Find time points from largest spatial deviations
    diffs = list()
    for p1, p2 in zip(pose_ours, pose_dlc):
        if p1 is None or p2 is None:
            diffs.append(0.0)
        else:
            p1 = np.reshape(np.array(p1), [-1, 3])
            p2 = np.reshape(np.array(p2), [-1, 3])
            diffs.append(
                np.mean(np.linalg.norm(p1[args.kp_id] - p2[args.kp_id], 2, -1))
            )
    diffs = np.array(diffs)
    # import matplotlib.pyplot as plt
    # plt.hist(diffs, bins=50)
    # plt.show()
    # sort_ind = np.argsort(diffs)
    sort_ind = np.argsort(diffs)[::-1]

    # pick trajectories
    traj, t_picked = list(), list()
    for t in sort_ind:  # pick the ones with the largest diff
        if len(t_picked) >= args.top_n:
            break
        if any([abs(t-x) < args.time*2 for x in t_picked]):
            continue

        t_picked.append(t)
        traj.append(
            [t-args.time, t+args.time]  # care about boundaries later on
        )

    N = int(0.25*len(sort_ind))
    for t in sort_ind[N:(N+args.top_n)]:  # pick some from the more erroneous quantile
        print(diffs[t])
        traj.append(
            [t-args.time, t+args.time]  # care about boundaries later on
        )

    # load input data
    video_list, K_list, dist_list, M_list = load(args.video_file, args.cam_wildcard)

    # iterate trajectories
    for traj_id, (fid1, fid2) in enumerate(traj):
        # randomly pick a camera
        cid = np.random.randint(0, len(video_list) - 1)

        # video reader
        reader = VideoReaderFast(video_list[cid],
                                 lambda x, K=K_list[cid], d=dist_list[cid]: preprocess(x, K, d, img_size=800))
        print('Opened video: %s with %d frames' % (video_list[cid], reader.get_size()))

        # video output name
        video_out_name = os.path.join(
            os.path.dirname(args.video_file),
            'traj_cmp%d.avi' % traj_id
        )
        writer = VideoWriterFast(video_out_name, 6.0, codec_t.divx, queue_size=256)
        writer.start()

        print('Showing trajectory', fid1, fid2)
        reader.set_fid(fid1)
        reader.start()

        for fid in range(fid1, fid2):
            img, K, img_shape = reader.read()

            # inpaint pose
            img_p = img.copy()
            for c, pose in zip(['g', 'b'],
                               [pose_ours, pose_dlc]):

                last_uv = None
                for t in range(5):
                    if fid-t < 0:
                        continue
                    if pose[fid-t] is None:
                        continue
                    p = np.reshape(np.array(pose[fid-t]), [-1, 3])
                    uv = cl.project(cl.trafo_coords(p, M_list[cid]), K)[args.kp_id]
                    # img_p = cv2.circle(img_p,
                    #                    (int(uv[0]), int(uv[1])),
                    #                    radius=2,
                    #                    color=(255, 0, 0) if c == 'b' else (0, 255, 0),
                    #                    thickness=-1)
                    col = (0, 255, 0) if c == 'g' else (0, 0, 255)

                    if last_uv is None:
                        img_p = cv2.circle(img_p,
                                           (int(uv[0]), int(uv[1])),
                                           radius=5,
                                           color=col,
                                           thickness=2)
                    else:
                        img_p = cv2.line(img_p,
                                         (int(uv[0]), int(uv[1])),
                                         (int(last_uv[0]), int(last_uv[1])),
                                         col, 2)
                    last_uv = uv

            writer.feed(img_p[:, :, ::-1])
            cv2.imshow('image', img_p[:, :, ::-1])
            cv2.waitKey(100)

        reader.stop()
        writer.wait_to_finish()
        writer.stop()



