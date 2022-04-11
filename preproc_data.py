from collections import defaultdict
from tqdm import tqdm
import argparse

from config.Model import Model

from utils.general_util import load_calib_data, json_load, json_dump, my_mkdir
from utils.index_util import *
from utils.triang import triangulate_robust


def process_labeled(model, cnt, out_path, db, anno, calib_all):
    """ How to process frames when there is annotation data available. """
    this_dataset_index_labeled = list()

    # find camera directories
    cam_base_dir = os.path.join(db['path'], db['frame_dir'])
    cam_names = ['cam%d' % cid for cid in db['cam_range']]
    cam_dirs = [os.path.join(cam_base_dir, x) for x in cam_names]
    print(' > Found %d cam directories in %s' % (len(cam_dirs), cam_base_dir))

    # find frame names
    frame_names = None
    for cd in cam_dirs:
        frames = list_frames(cd)
        print('\t> Found %d frames in %s' % (len(frames), cd))

        # check all cameras have the same amount of frames
        if frame_names is None:
            frame_names = [os.path.basename(x) for x in frames]
        assert len(frame_names) == len(frames), 'Number of frames does not match between cameras'

    min_num_kp = int(len(model.keypoints) * 0.8)
    # add to dataset index
    for f in tqdm(frame_names, desc=' > Processing %s' % db['path']):
        if f not in anno.keys():
            # this frame is not labeled
            continue

        # triangulate 2D points to 3D hypothesis
        K_list, _, M_list = calib_to_list(calib_all[-1], db['cam_range'])
        _, _, kp_uv, vis2d = anno_to_mat(anno[f], db['cam_range'], len(model.keypoints))
        points3d, _, vis3d, points2d_merged, vis_merged = triangulate_robust(kp_uv, vis2d, K_list, M_list)

        # sufficient number of 3d points found
        if np.sum(vis3d) < min_num_kp:
            continue

        img_c_list, scale_list, offset_list = preproc_sample(
            os.path.join(db['path'], db['frame_dir']),
            cam_names,
            f,
            points2d_merged,
            vis_merged,
            calib_all[-1],
            model.preprocessing['crop_oversampling'],
            model.preprocessing['crop_size']
        )

        for cam, img in zip(cam_names, img_c_list):
            tmp = os.path.join(out_path, cam, '%08d.jpg' % (cnt + len(this_dataset_index_labeled)))
            my_mkdir(tmp, is_file=True)
            cv2.imwrite(tmp, img)

        voxel_root = points3d[vis3d > 0.5].mean(0)
        this_dataset_index_labeled.append(
            [
                out_path,
                cam_names,
                '%08d.jpg' % (cnt + len(this_dataset_index_labeled)),
                points3d,
                vis3d,
                points2d_merged,
                vis_merged,
                voxel_root,
                scale_list,
                offset_list,
                len(calib_all) - 1
            ]
        )

    return this_dataset_index_labeled


def update_calib_id(index_data, offset):
    for i in range(len(index_data)):
        index_data[i][-1] += offset
    return index_data


def merge_all_index_files(model):
    # create output structures
    dataset_index_labeled, calib_all = defaultdict(list), list()

    # Iter all recordings and check which output files exist
    set_names = list()
    for db in model.datasets:
        ident = get_ident(db)
        out_path = os.path.join(model.preprocessing['data_storage'], ident)
        set_names.append(db['db_set'])

        # check for record file
        file_out_rec = os.path.join(out_path, model.preprocessing['index_file_name'] % db['db_set'])
        if os.path.exists(file_out_rec):
            data = json_load(file_out_rec)
            data = update_calib_id(data, len(calib_all))
            dataset_index_labeled[db['db_set']].extend(data)

        # calib file
        calib_file = os.path.join(out_path, model.preprocessing['calib_file'])
        if os.path.exists(calib_file):
            calib_all.extend(json_load(calib_file))

    # Save merged indices
    for set_name in set(set_names):
        file_out = model.preprocessing['index_file_name'] % set_name
        if len(dataset_index_labeled[set_name]) > 0:
            json_dump(os.path.join(
                model.preprocessing['data_storage'], file_out),
                dataset_index_labeled[set_name]
            )
            print('Saved %d samples to %s' % (len(dataset_index_labeled[set_name]),
                                              os.path.join(model.preprocessing['data_storage'], file_out)))

    # Save merged cam calibs
    if len(calib_all) > 0:
        json_dump(
            os.path.join(model.preprocessing['data_storage'], model.preprocessing['calib_file']),
            calib_all
        )


def preproc_data(model):
    """ Preprocess labeled data so we can train networks with it. """
    print('Running preprocessing for:', model)
    print('Saving to output folder:', model.preprocessing['data_storage'])

    # Init output structures
    calib_all = list()
    for i, db in enumerate(model.datasets):
        dataset_index = defaultdict(list)
        ident = get_ident(db)
        print('Preprocessing dataset entry %d: %s' % (i, ident))

        # where we want to save the processed frames
        output_path = os.path.join(model.preprocessing['data_storage'], ident)

        # check if we previously dealt with this record
        if os.path.exists(output_path):
            print(' > This record was already preprocessed previously.')
            continue

        # check base paths existance
        if not os.path.exists(db['path']):
            print(' > Base path not found: %s' % db['path'])
            continue

        # check calib file
        calib_file_path = os.path.join(db['path'], db['calib'])
        if not os.path.exists(calib_file_path):
            print(' > Calib file not found: %s' % calib_file_path)
            continue
        calib_all.append(load_calib_data(calib_file_path, return_cam2world=False))

        # check annotation file
        anno_file = os.path.join(db['path'], db['frame_dir'], db['anno'])
        if os.path.exists(anno_file):
            print(' > Loading annotations from %s' % anno_file)
            anno = json_load(anno_file)
            print(' > Got %d annotations' % len(anno))

        else:
            print(' > Cant find annotation file: %s' % anno_file)
            print(' > Assuming dataset is not labeled.')
            continue

        if check_if_labeled(anno):
            print(' > Found labeled sequence: %s' % os.path.join(db['path'], db['frame_dir']))
            cnt = sum([len(x) for x in dataset_index.values()])
            this_index = process_labeled(model, cnt, output_path, db, anno, calib_all)
            print(' > Adding %d samples to labeled set %s' % (len(this_index), db['db_set']))
            dataset_index[db['db_set']].extend(
                this_index
            )
        else:
            print(' > Sequence appears to be unlabeled (f.e. annotation file is empty).')

        if len(dataset_index[db['db_set']]) > 0:
            file_out_rec = os.path.join(output_path, model.preprocessing['index_file_name'] % db['db_set'])
            json_dump(file_out_rec, dataset_index[db['db_set']])
            print(' > Saved %d samples to %s' % (len(dataset_index[db['db_set']]), file_out_rec))

            # save Calib file
            json_dump(
                os.path.join(output_path, model.preprocessing['calib_file']),
                calib_all
            )

    merge_all_index_files(model)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Preprocess data for network training.')
    parser.add_argument('model', type=str, help='Model definition file.')
    args = parser.parse_args()

    m = Model(args.model)
    preproc_data(m)
