from __future__ import print_function, unicode_literals
import os
import json
import tensorpack as tp
import time
import numpy as np

from utils.general_util import json_load
from utils.InfiniteShuffleVideoReader import InfiniteShuffleVideoReaderFast


""" Meta class representing one data point and some functions attatched to it. """
class DatasetLabeledMeta:
    def __init__(self, base_path, cam_dirs, frame_name,
                 xyz, vis, uv_merged, vis_merged,
                 voxel_root, scales, offsets, calib_id, calib_path):
        self.img_paths = [os.path.join(base_path, c, frame_name) for c in cam_dirs]
        self.xyz = np.array(xyz)
        self.vis = np.array(vis)
        self.uv_merged = np.array(uv_merged)
        self.vis_merged = np.array(vis_merged)
        self.voxel_root = np.array(voxel_root)
        self.scales = np.array(scales)
        self.offsets = np.array(offsets)
        self.calib_id = calib_id
        self.calib_path = calib_path
        self.cam_range = cam_dirs

        self.img_list = list()  # this contains the images
        self.M = list()  # intrinsics
        self.K = list()  # extrinsics


class DatasetUnlabeledMeta:
    def __init__(self, video_set, calib, cam_range, pred_bb_file):
        self.video_set = video_set
        self.calib = calib
        self.cam_range = ['cam%d' % i for i in cam_range]
        self.pred_bb_file = pred_bb_file


# This thing know how to iterate over the dataset, how large it is and so on
class DatasetSemiLabeled(tp.DataFlow):
    @staticmethod
    def get_samples(model, dataset_type):
        index_file = os.path.join(model.preprocessing['data_storage'], model.preprocessing['index_file_name'] % dataset_type)
        calib_file = os.path.join(model.preprocessing['data_storage'], model.preprocessing['calib_file'])
        msg = 'Index file not found: %s' % index_file
        assert os.path.exists(index_file), msg

        # load data
        with open(index_file, 'r') as fi:
            data = json.load(fi)

        # insert calib_path to the data, not a nice solution
        data_new = list()
        for d in data:
            tmp = list(d)
            tmp.append(calib_file)
            data_new.append(tmp)
        data = data_new

        msg = 'Dataset appears to be empty: %s' % dataset_type
        assert len(data) > 0, msg

        return data

    def __init__(self, model, dataset_subset_list, video_name_template_list, pred_bb_list, single_sample=False, shuffle=True):
        """ Gets a list of dataset subset types which creates a generator that will evenly sample from the given set. """
        super(DatasetSemiLabeled, self).__init__()
        self.anno_list = list()
        self.source_id = -1
        self.frame_id = dict()
        self.single_sample = single_sample
        self.shuffle = shuffle
        self._idx = 0

        # accumulate index for all samples
        print('Building dataset index ...')
        start = time.time()

        self.num_samples = 0
        self.max_iterations = 0
        for i, dataset in enumerate(dataset_subset_list):
            data = self.get_samples(model, dataset)
            num_samples = len(data)
            self.anno_list.append(data)

            self.frame_id[i] = -1
            self.num_samples += num_samples
            self.max_iterations = max(self.max_iterations, num_samples)

        print('Index building done in %.1f sec' % (time.time() - start))
        print('Dataset yields %d samples from %d datasets %s.'
              ' We iterate for %d steps' % (self.num_samples,
                                            len(dataset_subset_list),
                                            dataset_subset_list,
                                            self.max_iterations))

        # Search for videos that match this template name
        video_set_list, calib_list, cam_ranges_list = list(), list(), list()
        for video_name_template in video_name_template_list:
            tmp, tmp2 = list(), list()
            for cid in range(10):
                if os.path.exists(video_name_template % cid):
                    tmp.append(video_name_template % cid)
                    tmp2.append(cid)

            if len(tmp) > 0:
                video_set_list.append(tmp)
                cam_ranges_list.append(tmp2)
                calib_list.append(
                    os.path.join(os.path.dirname(tmp[0]), 'M.json')
                )

        self.unlabeled_samples = list(zip(video_set_list, calib_list, cam_ranges_list, pred_bb_list))

        print('Got additionally %d unlabeled sequences:' % len(self.unlabeled_samples))
        for vid_sets, _, cr, _ in self.unlabeled_samples:
            print(vid_sets, ' in ', cr)

    def reset_state(self):
        # reset counters
        self.source_id = -1
        self._idx = 0

        if self.single_sample:
            # dont do the reshuffling because we want only a single sample anyways
            return

        if not self.shuffle:
            # dont do the reshuffling
            return

        np.random.seed()  # set a new random seed

        # randomly shuffle order of datasets
        ind_rnd = np.random.permutation(len(self.anno_list))
        self.anno_list = [self.anno_list[i] for i in ind_rnd]

        # randomly shuffle order within each dataset
        for i in range(len(self.anno_list)):
            ind_rnd = np.random.permutation(len(self.anno_list[i]))

            self.anno_list[i] = [self.anno_list[i][j] for j in ind_rnd]

    def size(self):
        return self.max_iterations

    def get_data(self):
        uid = 0
        while True:
            if self._idx >= self.size():
                self._idx = 0
            else:
                self._idx += 1

            self.source_id = (self.source_id + 1) % len(self.anno_list)
            self.frame_id[self.source_id] = (self.frame_id[self.source_id] + 1) % len(self.anno_list[self.source_id])
            sid = self.source_id
            fid = self.frame_id[self.source_id]

            uid = (uid + 1) % len(self.unlabeled_samples)

            if self.single_sample:
                yield [DatasetLabeledMeta(*self.anno_list[0][0])]
                yield [DatasetUnlabeledMeta(*self.unlabeled_samples[0])]
            else:
                yield [DatasetLabeledMeta(*self.anno_list[sid][fid])]
                yield [DatasetUnlabeledMeta(*self.unlabeled_samples[uid])]


