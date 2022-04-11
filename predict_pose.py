import argparse, os
import numpy as np
from tqdm import tqdm
import cv2
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from config.Model import Model
from config.Param import Param
from utils.general_util import find_last_existant, json_load, json_dump, compensate_crop_K, preprocess_image
from utils.VideoReaderFast import VideoReaderFast
from utils.plot_util import draw_skel, draw_bb
from utils.StitchedImage import StitchedImage

from pose_network.core.Types import *
from pose_network.load_util import load_pose_network
import utils.CamLib as cl

from predict_bb import parse_input, preprocess


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Show labeled datasets.')
    parser.add_argument('model', type=str, help='Model definition file.')
    parser.add_argument('video', type=str, help='Video file.')
    parser.add_argument('--show', action='store_true', help='Show prediction.')
    parser.add_argument('--use_refine', action='store_true', help='Activate refinement module.')
    parser.add_argument('--cam_wildcard', type=str, default='cam%d', help='How to tell the camera id'
                                                                          ' from a given file name.')

    parser.add_argument('--run_wildcard', type=str, default='run%03d', help='How to tell the run id'
                                                                            ' from a given file name.')
    parser.add_argument('--max_cams', type=int, default=64, help='Maximal number of cams we search for.')
    parser.add_argument('--calib_file_name', type=str, default='M.json', help='Assumed calibration file name.')
    parser.add_argument('--frame', type=int, help='Show one dedicated frame only')
    args = parser.parse_args()

    # load model data
    model = Model(args.model)
    param = Param()

    # setup up pose network
    sess, input_tensors, pred_tensors = load_pose_network(param.pose, model)

    # parse given input
    video_list, K_list, \
    dist_list, M_list,\
    pred_file_name = parse_input(args.video, args.cam_wildcard, args.run_wildcard, args.max_cams, args.calib_file_name,
                                 find_last_existant)
    print('Found %s video files to make predictions: %s' % (len(video_list), video_list[0]))
    print('Predictions will be saved to: %s' % pred_file_name)

    # load bb annotations
    predictions = json_load(pred_file_name)

    # create video readers
    video_readers = [VideoReaderFast(v, lambda x,K=K: preprocess(x, K, img_size=800)) for v, K, dist in zip(video_list, K_list, dist_list)]

    # start them
    for v in video_readers:
        if args.frame is not None:
            v.set_fid(args.frame)
        else:
            args.frame = 0
        v.start()

    for idx in tqdm(range(args.frame, video_readers[0].get_size()), desc='Predicting'):
        # read frame
        data = [v.read() for v in video_readers]
        img_list, K_list = [d[0] for d in data], [d[1] for d in data]
        imgs = np.stack(img_list, 0)
        K_list = np.stack(K_list, 0)

        if predictions[idx]['xyz'] is None:
            print('%s doesnt have a root point. Skipping it.' % idx)
            continue

        # get bb prediction
        root, boxes = np.array(predictions[idx]['xyz']), predictions[idx]['boxes']

        img_c, K_c = list(), list()
        for i, (box, img) in enumerate(zip(boxes, imgs)):
            h, w = img.shape[:2]

            if np.all(np.abs(box) < 1e-4):
                # if not detected use full image
                box = [0.0, 1.0, 0.0, 1.0]

            box_scales = np.array([[box[2] * h, box[0] * w],
                                   [box[3] * h, box[1] * w]])
            img_crop, scale, offset, img_raw_crop = preprocess_image(img, box_scales,
                                                                     do_mean_subtraction=True,
                                                                     over_sampling=model.preprocessing['crop_oversampling'],
                                                                     symmetric=True,
                                                                     resize=True,
                                                                     target_size=model.preprocessing['crop_size'])

            img_c.append(img_raw_crop.astype(np.uint8))
            K_c.append(compensate_crop_K(K_list[i], scale, offset))

        if args.use_refine:
            xyz_tf = pred_tensors[data_t.pred_xyz_refine]
            uv_tf = pred_tensors[data_t.pred_uv_refine][-1]
        else:
            xyz_tf = pred_tensors[data_t.pred_xyz][-1]
            uv_tf = pred_tensors[data_t.pred_uv_final]

        # img_input = np.stack(img_c, 0).astype(np.float32) / 255.0 - 0.5
        img_input = np.stack(img_c, 0)[:, :, :, ::-1].astype(np.float32) / 255.0 - 0.5  # make BGR and subtract rough mean

        feeds = {
            input_tensors['image']: img_input,
            input_tensors['K']: np.stack(K_c, 0),
            input_tensors['M']: np.stack(M_list, 0),
            input_tensors['root']: np.expand_dims(root, 0)
        }
        fetches = [
            xyz_tf,
            uv_tf,
            pred_tensors[data_t.pred_score3d][-1]
        ]
        fetches_v = sess.run(fetches, feed_dict=feeds)

        # save predictions
        xyz_pred, uv_pred, score_pred = fetches_v
        predictions[idx]['kp_xyz'] = xyz_pred
        predictions[idx]['kp_score'] = score_pred

        if args.show:
            img_list = list()
            for i, (this_img, this_box, this_uv) in enumerate(zip(imgs, boxes, uv_pred)):
                uv_proj = cl.project(cl.trafo_coords(xyz_pred[0], M_list[i]), K_list[i])
                h, w = this_img.shape[:2]
                if np.all(np.abs(this_box) < 1e-4):
                    # if not detected use full image
                    this_box = [0.0, 1.0, 0.0, 1.0]
                box_scaled = np.array([this_box[0] * w, this_box[1] * w, this_box[2] * h, this_box[3] * h])
                this_img_box = draw_bb(this_img, box_scaled, mode='lrtb', color='g', linewidth=2)

                root_uv = cl.project(cl.trafo_coords(root, M_list[i]), K_list[i])
                this_img_box = cv2.circle(this_img_box,
                                          (int(root_uv[0, 0]), int(root_uv[0, 1])),
                                          radius=5,
                                          color=(0, 255, 255),
                                          thickness=-1)

                img_list.append(
                    draw_skel(this_img_box, model, uv_proj, order='uv', linewidth=2, kp_style=(5, 1))
                )

            merge = StitchedImage(img_list)
            cv2.imshow('pose pred', merge.image[:, :, ::-1])

            if args.frame is not None:
                cv2.waitKey()
            else:
                cv2.waitKey(10)

    json_dump(pred_file_name, predictions, verbose=True)
