import argparse, os
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from tqdm import tqdm

from config.Model import Model
from utils.general_util import load_calib_data, compensate_crop_K, json_dump, find_first_non_existant, parse_file_name
from utils.VideoReaderFast import VideoReaderFast
from utils.detector_util import *
from utils.plot_util import draw_bb
from utils.StitchedImage import StitchedImage


def parse_input(video_file_path, cam_template, run_template, max_num, calib_file_name, name_fct):
    assert os.path.exists(video_file_path), 'Video file not found.'
    assert os.path.isfile(video_file_path), 'Assumes a path to a video file is given, not a directory).'

    # split into parts
    video_path = os.path.dirname(video_file_path)
    video_name = os.path.basename(video_file_path)

    # try to find calibration file
    calib_file_name = os.path.join(video_path, calib_file_name)
    assert os.path.exists(calib_file_name), 'Calibration file not found.'

    # find out camera id of the given video (could be any)
    _, run_id, given_cid = parse_file_name(video_file_path, run_template, cam_template)
    assert given_cid is not None, 'Given camera template was not found in the given video name.'

    # find available cams
    record_name = None
    video_list, cam_range = list(), list()
    for cid in range(max_num):
        if record_name is None:
            record_name = os.path.splitext(video_name)[0]
            record_name = record_name.replace(cam_template % given_cid, '')

        test_path = video_file_path.replace(
            cam_template % given_cid,
            cam_template % cid
        )
        if os.path.exists(test_path):
            video_list.append(test_path)
            cam_range.append(cid)

    # load and check calibration
    calib = load_calib_data(calib_file_name, return_cam2world=False)
    assert all(['cam%d' % cid in calib.keys() for cid in cam_range]), 'Missing calibration data for at least one camera.'

    # turn into lists
    K_list = [np.array(calib['cam%d' % i]['K']) for i in cam_range]
    dist_list = [np.array(calib['cam%d' % i]['dist']) for i in cam_range]
    M_list = [np.array(calib['cam%d' % i]['M']) for i in cam_range]

    # output file name
    pred_out_name = None
    if name_fct is not None:
        pred_out_name = name_fct(os.path.join(video_path, 'pred_%s_%%02d.json' % record_name))
        assert pred_out_name is not None, 'Could not deduct valid prediction file.'

    return video_list, K_list, dist_list, M_list, pred_out_name


def preprocess(frame, K, dist=None, img_size=224):
    K = K.copy()
    if dist is not None:
        dist = dist.copy()
        frame = cv2.undistort(frame, K, dist)
    # make image square
    s = np.array(frame.shape[:2]) / np.array([img_size, img_size], dtype=np.float32)
    frame_c = cv2.resize(frame, (img_size, img_size))

    # # keep aspect ratio
    # s = np.max(np.array(frame.shape[:2]) / np.array([img_size, img_size], dtype=np.float32))
    # img_sizes = np.array(frame.shape[:2], dtype=np.float32)/s
    # img_sizes = np.round(img_sizes).astype(np.int32)
    # frame_c = cv2.resize(frame, (img_sizes[1], img_sizes[0]))
    # s = np.array(frame.shape[:2]) / img_sizes.astype(np.float32)  # effective size

    frame_c = frame_c[:, :, ::-1]  # make RGB
    K_c = compensate_crop_K(K, s, (0, 0))
    return frame_c, K_c, frame.shape[:2]


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Show labeled datasets.')
    parser.add_argument('model', type=str, help='Model definition file.')
    parser.add_argument('video', type=str, help='Video file.')
    parser.add_argument('--show', action='store_true', help='Show prediction.')
    parser.add_argument('--cam_wildcard', type=str, default='cam%d', help='How to tell the camera id'
                                                                          ' from a given file name.')

    parser.add_argument('--run_wildcard', type=str, default='run%03d', help='How to tell the run id'
                                                                            ' from a given file name.')
    parser.add_argument('--max_cams', type=int, default=64, help='Maximal number of cams we search for.')
    parser.add_argument('--calib_file_name', type=str, default='M.json', help='Assumed calibration file name.')
    args = parser.parse_args()

    # load model data
    model = Model(args.model)

    # parse given input
    video_list, K_list, \
    dist_list, M_list,\
    pred_file_name = parse_input(args.video, args.cam_wildcard, args.run_wildcard, args.max_cams, args.calib_file_name,
                                 find_first_non_existant)
    print('Found %s video files to make predictions: %s' % (len(video_list), video_list[0]))
    print('Predictions will be saved to: %s' % pred_file_name)

    if model.preprocessing['bb_fixed'] is None:
        # build network graph
        print('Loading BB network model: %s' % model.bb_models[-1])
        sess, input_tensors, pred_tensors = load_inference_graph(model.bb_models[-1])
    else:
        assert len(model.preprocessing['bb_fixed']) == len(video_list), 'There has to be one bounding box specified for each video.'

    # create video readers
    video_readers = [VideoReaderFast(v, lambda x,K=K: preprocess(x, K)) for v, K in zip(video_list, K_list)]

    # start them
    for v in video_readers:
        v.start()

    predictions = list()
    for idx in tqdm(range(video_readers[0].get_size()), desc='Predicting'):
        # read frame
        data = [v.read() for v in video_readers]
        img_list, K_list, orig_shapes = [d[0] for d in data], [d[1] for d in data], [d[2] for d in data]
        imgs = np.stack(img_list, 0)
        orig_shapes = np.stack(orig_shapes, 0)

        if model.preprocessing['bb_fixed'] is None:
            # pass through network
            boxes, scores,  = sess.run(
                [pred_tensors['detection_boxes'],
                 pred_tensors['detection_scores']],
                feed_dict={input_tensors['image_tensor']: imgs})
        else:
            boxes = np.array(model.preprocessing['bb_fixed'], dtype=np.float32)
            # boxes = boxes_tmp.copy()
            # boxes[:, 0] = boxes_tmp[:, 1]
            # boxes[:, 1] = boxes_tmp[:, 0]
            # boxes[:, 3] = boxes_tmp[:, 2]
            # boxes[:, 2] = boxes_tmp[:, 3]

            boxes[:, 1] /= orig_shapes[:, 1]
            boxes[:, 3] /= orig_shapes[:, 1]
            boxes[:, 0] /= orig_shapes[:, 0]
            boxes[:, 2] /= orig_shapes[:, 0]
            boxes = np.expand_dims(boxes, 1)
            scores = np.ones_like(boxes[:, :, 0])

        # process boxes
        pred = post_process_detections(boxes, scores,
                                       K_list,
                                       M_list,
                                       imgs.shape[1:3],
                                       verbose=False)
        predictions.append(pred)

        if args.show:
            img_vis_list = list()
            for bid in range(imgs.shape[0]):
                root_uv = cl.project(cl.trafo_coords(pred['xyz'], M_list[bid]), K_list[bid])
                img = cv2.circle(imgs[bid].astype(np.uint8),
                                 (int(root_uv[0, 0]), int(root_uv[0, 1])),
                                 radius=5,
                                 color=(0, 255, 255),
                                 thickness=-1)
                img_vis_list.append(draw_bb(img,
                                            pred['boxes'][bid] * imgs.shape[1],
                                            mode='lrtb', color='g'))

            merge = StitchedImage(img_vis_list)
            cv2.imshow('img_bb_post', merge.image[:, :, ::-1])
            cv2.waitKey(100)

    json_dump(pred_file_name, predictions, verbose=True)
