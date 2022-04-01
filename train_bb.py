from __future__ import print_function, unicode_literals
import os
os.environ['PYTHONPATH'] = '%s' % os.path.join(os.getcwd(), '3rd_party')  # add 3rd party libraries
import tensorflow as tf
import numpy as np
import random, argparse
from tqdm import tqdm

from config.Model import Model
from config.Param import Param
from utils.general_util import load_calib_data, json_load
from utils.index_util import calib_to_list, anno_to_mat, check_if_labeled
from utils.triang import triangulate_robust
from utils.general_util import calc_bbox, get_img_size, find_first_non_existant
from bb_network.write_config import write


def _dump_bb_cvs(filename, data):
    print('Dumping %d items to %s' % (len(data), filename))

    with open(filename, 'w') as fo:
        # Header
        fo.write('filename,width,height,class,xmin,ymin,xmax,ymax\n')

        # Body
        for item in data:
            fo.write('%s,%d,%d,%s,%d,%d,%d,%d\n' % tuple(item))


def write_bb_csv(model, param):
    all_data = list()

    # Iterate over listed data
    for db in model.datasets:
        # check base paths existance
        msg = 'Base path not found: %s' % db['path']
        assert os.path.exists(db['path']), msg
        print('Dealing with: %s' % db['path'])

        # check calib file
        calib_file_path = os.path.join(db['path'], db['calib'])
        msg = 'Calib file not found: %s' % calib_file_path
        assert os.path.exists(calib_file_path), msg
        this_calib = load_calib_data(calib_file_path, return_cam2world=False)
        print('Located associated calibration file: %s' % calib_file_path)

        # check annotation file
        anno = dict()
        anno_file = os.path.join(db['path'], db['frame_dir'], db['anno'])
        if os.path.exists(anno_file):
            print('Loading annotations from %s' % anno_file)
            anno = json_load(anno_file)
        print('Got %d annotations' % len(anno))

        # figure out if we want to add it to the labeled subset
        kp_thresh = round(len(model.keypoints) * 0.8)
        if check_if_labeled(anno, num_kp_thresh=kp_thresh):
            cam_names = ['cam%d' % cid for cid in db['cam_range']]
            for f, anno_frame in tqdm(anno.items(), desc='Processing annotations'):

                K_list, _, M_list = calib_to_list(this_calib, db['cam_range'])
                _, _, kp_uv, vis2d = anno_to_mat(anno_frame, db['cam_range'], len(model.keypoints))
                _, _, vis3d, points2d_merged, vis_merged = triangulate_robust(kp_uv, vis2d, K_list, M_list)

                if np.sum(vis3d) < 0.8*len(model.keypoints):
                    # skip if too little 3D point are visible
                    continue

                for i, cam in enumerate(cam_names):
                    img_path = os.path.join(db['path'], db['frame_dir'], cam, f)
                    w, h = get_img_size(img_path)
                    bbox = calc_bbox(points2d_merged[i], vis_merged[i])

                    # import cv2
                    # from utils.plot_util_rat import draw_bb
                    # img = cv2.imread(img_path)
                    # box_ltrb = bbox[0, 1], bbox[0, 0], bbox[1, 1], bbox[1, 0]
                    # img = draw_bb(img, box_ltrb, linewidth=3)
                    # cv2.imshow('img+box', img)
                    # cv2.waitKey()

                    entry = [
                        os.path.abspath(img_path),
                        w, h,
                        'object',
                        bbox[0, 1], bbox[0, 0], bbox[1, 1], bbox[1, 0]
                    ]
                    all_data.append(entry)

    assert len(all_data) > 0, "No samples available to train on."

    # split train/test
    random.shuffle(all_data)
    N = int(len(all_data)*0.8)
    all_data_train = all_data[:N]
    all_data_test = all_data[N:]

    # dump to file
    csv_out_train = os.path.join(model.preprocessing['data_storage'], param.bb['detector_csv_train'])
    _dump_bb_cvs(csv_out_train,
                 all_data_train)
    csv_out_test = os.path.join(model.preprocessing['data_storage'], param.bb['detector_csv_test'])
    _dump_bb_cvs(csv_out_test,
                 all_data_test)
    return csv_out_train, csv_out_test


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Start training the bounding box estimation network.')
    parser.add_argument('model', type=str, help='Model definition file.')
    args = parser.parse_args()

    model = Model(args.model)
    param = Param()

    # complete paths
    bb_job_name = find_first_non_existant(os.path.join('./trainings', param.bb['bb_job_name']))
    bb_config_file = os.path.join(os.getcwd(), param.bb['bb_config_file'])
    bb_train_dataset = os.path.join(model.preprocessing['data_storage'], param.bb['bb_train_dataset'])
    bb_test_dataset = os.path.join(model.preprocessing['data_storage'], param.bb['bb_test_dataset'])

    # write labels to csv file
    csv_file_train, csv_file_test = write_bb_csv(model, param)

    # generate tf record
    os.system('python bb_network/generate_tfrecord.py --csv_input=%s  --output_path=%s' % (os.path.abspath(csv_file_train),
                                                                                           os.path.abspath(bb_train_dataset)))
    os.system('python bb_network/generate_tfrecord.py --csv_input=%s  --output_path=%s' % (os.path.abspath(csv_file_test),
                                                                                           os.path.abspath(bb_test_dataset)))

    # Write config file for training
    write(param.bb, bb_config_file, bb_train_dataset, bb_test_dataset, 'mobilenet')

    # run training of bb estimation network
    os.system('python bb_network/model_main.py --logtostderr --model_dir=%s --pipeline_config_path=%s' % (bb_job_name,
                                                                                                          bb_config_file))

    # # find last checkpoint
    last_ckpt = tf.train.latest_checkpoint(bb_job_name)
    assert last_ckpt is not None, "No snapshot found!"

    # export graph
    os.system('python bb_network/export_inference_graph.py --input_type image_tensor'
              ' --pipeline_config_path %s --trained_checkpoint_prefix %s --output_directory %s' % (bb_config_file,
                                                                                                   last_ckpt,
                                                                                                   bb_job_name + '_graph'))

