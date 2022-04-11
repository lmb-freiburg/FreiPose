import argparse, os
from tqdm import tqdm
import numpy as np
import cv2

from config.Model import Model
from config.Param import Param
from predict_bb import parse_input, preprocess
from utils.general_util import find_last_existant, json_load
from utils.plot_util import draw_skel, draw_bb, draw_text
from utils.VideoReaderFast import VideoReaderFast
from utils.VideoWriterFast import *
from utils.StitchedImage import StitchedImage
import utils.CamLib as cl

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Show predictions.')
    parser.add_argument('model', type=str, help='Model definition file.')
    parser.add_argument('video', type=str, help='Input video file.')
    parser.add_argument('--pred_file', type=str, help='Predcition file to visualize.')
    parser.add_argument('--hide_bb', action='store_true', help='Not show bounding box prediction.')
    parser.add_argument('--draw_root', action='store_true', help='Show voxel root prediction.')
    parser.add_argument('--draw_fid', action='store_true', help='Inpaint the frame id into the video.')
    parser.add_argument('--save', action='store_true', help='Save prediction to a video file.')
    parser.add_argument('--wait', action='store_true', help='Wait after each frame shown.')

    parser.add_argument('--run_wildcard', type=str, default='run%03d', help='How to tell the run id'
                                                                            ' from a given file name.')
    parser.add_argument('--cam_wildcard', type=str, default='cam%d', help='How to tell the camera id'
                                                                          ' from a given file name.')
    parser.add_argument('--max_cams', type=int, default=64, help='Maximal number of cams we search for.')
    parser.add_argument('--window_size', type=int, default=1200, help='Window used for visualization.')
    parser.add_argument('--calib_file_name', type=str, default='M.json', help='Assumed calibration file name.')
    parser.add_argument('--start_fid', type=int, help='Starting frame id.')
    args = parser.parse_args()

    # load model data
    model = Model(args.model)
    param = Param()

    # parse given input
    video_list, K_list, \
    dist_list, M_list,\
    pred_file_name = parse_input(args.video,
                                 args.cam_wildcard, args.run_wildcard, args.max_cams,
                                 args.calib_file_name,
                                 find_last_existant)
    print('Found %s video files to make predictions: %s' % (len(video_list), video_list[0]))
    print('Predictions will be saved to: %s' % pred_file_name)

    # load bb annotations
    if args.pred_file is not None:
        pred_file_name = args.pred_file
    predictions = json_load(pred_file_name)

    # create video readers
    video_readers = [VideoReaderFast(v, lambda x, K=K: preprocess(x, K, img_size=800)) for v, K, dist in
                     zip(video_list, K_list, dist_list)]

    # start them
    for v in video_readers:
        if args.start_fid is not None:
            v.set_fid(args.start_fid)
        else:
            args.start_fid = 0
        v.start()

    if args.save:
        # video output name
        video_out_name = pred_file_name.replace('/pred_', '/vid_pred_').replace('.json', '.avi')
        writer = VideoWriterFast(video_out_name, 6.0, codec_t.divx, queue_size=256)
        writer.start()

    for idx in tqdm(range(args.start_fid, video_readers[0].get_size()), desc='Showing'):
        # read frame
        data = [v.read() for v in video_readers]
        img_list, K_list = [d[0] for d in data], [d[1] for d in data]
        imgs = np.stack(img_list, 0)
        K_list = np.stack(K_list, 0)

        img_list = list()
        for i, (this_img, this_box) in enumerate(zip(imgs, predictions[idx]['boxes'])):

            if not args.hide_bb:
                # draw bounding box
                h, w = this_img.shape[:2]
                if np.all(np.abs(this_box) < 1e-4):
                    # if not detected use full image
                    this_box = [0.0, 1.0, 0.0, 1.0]
                box_scaled = np.array([this_box[0] * w, this_box[1] * w, this_box[2] * h, this_box[3] * h])
                this_img = draw_bb(this_img, box_scaled, mode='lrtb', color='g', linewidth=2)

            if args.draw_root:
                # draw voxel root
                root_uv = cl.project(cl.trafo_coords(np.array(predictions[idx]['xyz']), M_list[i]), K_list[i])
                this_img = cv2.circle(this_img,
                                      (int(root_uv[0, 0]), int(root_uv[0, 1])),
                                      radius=5,
                                      color=(0, 255, 255),
                                      thickness=-1)

            # draw keypoints
            if 'kp_xyz' in predictions[idx].keys():
                uv_proj = cl.project(cl.trafo_coords(np.array(predictions[idx]['kp_xyz'][0]), M_list[i]), K_list[i])
                this_img = draw_skel(this_img, model, uv_proj, order='uv', linewidth=2, kp_style=(5, 1))

            # draw frame id
            if args.draw_fid and i == 0:
                this_img = draw_text(this_img, '%03d' % idx)

            img_list.append(this_img)

        merge = StitchedImage(img_list, target_size=(int(0.8*args.window_size), args.window_size))

        if args.save:
            writer.feed(merge.image[:, :, ::-1])

        cv2.imshow('pose pred', merge.image[:, :, ::-1])
        cv2.waitKey(0 if args.wait else 10)

    # end readers
    for v in video_readers:
        v.stop()

    # end writer
    if args.save:
        writer.wait_to_finish()
        writer.stop()