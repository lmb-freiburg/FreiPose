from __future__ import print_function, unicode_literals
import tensorpack as tp
import random
import cv2
import sys
sys.path.append('/home/zimmermc/projects/ALTool')

from utils.general_util import compensate_crop_coords, compensate_crop_K
import utils.CamLib as cl
from pose_network.core.Types import *
from datareader.DatasetSemiLabeled import *

from utils.general_util import json_load, compensate_crop_K, preprocess_image

g_voxel_dim, g_voxel_resolution = 64, 0.4


def read_vid_length(video_path):
    """ Reads a single frame from a video.
    """
    cap = cv2.VideoCapture(video_path)
    vid_size = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    return vid_size


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


def read_random_video_frames(video_list):
    fid = random.randint(0, read_vid_length(video_list[0])-1)
    return fid, [read_vid_frame(x, fid) for x in video_list]


def build_dataflow(model, datasets_list,
                   vid_list, pred_bb_list,
                   is_train=False, threaded=False, shuffle=True,
                   single_sample=False,
                   voxel_dim=64, voxel_resolution=0.4):
    global g_voxel_dim, g_voxel_resolution
    g_voxel_dim, g_voxel_resolution = voxel_dim, voxel_resolution

    # build dataflow
    ds = DatasetSemiLabeled(model, datasets_list,
                            vid_list, pred_bb_list,
                            single_sample, shuffle=shuffle)
    if threaded:
        ds = tp.MultiThreadMapData(ds, map_func=read_img, nr_thread=8, buffer_size=16)
    else:
        ds = tp.MapData(ds, read_img)

    if is_train:
        ds = tp.MapDataComponent(ds, augment_hue)
        ds = tp.MapDataComponent(ds, augment_snp_noise)
        ds = tp.MapDataComponent(ds, augment_blur)
        ds = tp.MapDataComponent(ds, augment_root)
        # pass

    ds = tp.MapData(ds, func=pose_to_img)
    ds.reset_state()
    return ds


calib_cache = None
def _get_calib(calib_id, calib_path):
    global calib_cache

    if calib_cache is None:
        # load calib
        calib_cache = json_load(calib_path)

    return calib_cache[calib_id]

calib_cache_videos = dict()
def _get_calib_videos(calib_path):
    global calib_cache_videos

    if calib_path not in calib_cache_videos.keys():
        # load calib
        calib_cache_videos[calib_path] = json_load(calib_path)

    return calib_cache_videos[calib_path]

predictions_bb_cache = dict()
def _get_pred_bb(pred_file, idx):
    global predictions_bb_cache

    if pred_file not in predictions_bb_cache.keys():
        # load calib
        predictions_bb_cache[pred_file] = json_load(pred_file)

    return predictions_bb_cache[pred_file][idx]['boxes'], predictions_bb_cache[pred_file][idx]['xyz']


def _crop_images(imgs, boxes, K_list):
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
                                                                 over_sampling=2.0,
                                                                 symmetric=True,
                                                                 resize=True,
                                                                 target_size=224)

        img_c.append(img_raw_crop.astype(np.float32))
        K_c.append(compensate_crop_K(K_list[i], scale, offset))
    return img_c, K_c


def read_img(metas):
    for meta in metas:
        if type(meta) == DatasetLabeledMeta:
            if not all([os.path.exists(x) for x in meta.img_paths]):
                print('Path not found: ', meta.img_paths[0])

            meta.is_supervised = 1.0

            # read all images
            meta.img_list = [cv2.imread(x).astype(np.float32) for x in meta.img_paths]

            # read calibration
            calib = _get_calib(meta.calib_id, meta.calib_path)
            # dist = [np.array(calib[cam]['dist']) for cam in meta.cam_range]
            meta.K_list = [np.array(calib[cam]['K']) for cam in meta.cam_range]
            meta.M_list = [np.array(calib[cam]['M']) for cam in meta.cam_range]
            # meta.M_list = [np.linalg.inv(M) for M in meta.M_list]

            # compensate for crop
            meta.K_list = [compensate_crop_K(K, s, o) for K, s, o in zip(meta.K_list, meta.scales, meta.offsets)]

            # compensate 2D coordinates for crop
            meta.uv_merged = [compensate_crop_coords(pts, s, o) for pts, s, o in zip(meta.uv_merged, meta.scales, meta.offsets)]

            # # undistort them
            # meta.img_list = [cv2.undistort(I, K, dist) for I, K, dist in zip(meta.img_list, meta.K_list, dist)]

            meta.uv = [cl.project(cl.trafo_coords(meta.xyz, M), K) for K, M in zip(meta.K_list, meta.M_list)]

        elif type(meta) == DatasetUnlabeledMeta:

            meta.is_supervised = 0.0

            # read a video frames
            fid, meta.img_list = read_random_video_frames(meta.video_set)

            # read calibration
            calib = _get_calib_videos(meta.calib)
            # dist = [np.array(calib[cam]['dist']) for cam in meta.cam_range]
            meta.K_list = [np.array(calib['K'][cam]) for cam in meta.cam_range]
            meta.M_list = [np.array(calib['M'][cam]) for cam in meta.cam_range]

            # read bounding box
            boxes, meta.voxel_root = _get_pred_bb(meta.pred_bb_file, fid)

            if meta.voxel_root is None:
                print('Invalid root in unlabeled sequence, Skipping.')
                return None

            # crop images according to the boxes
            meta.img_list, meta.K_list = _crop_images(meta.img_list, boxes, meta.K_list)

            # create dummy data just so its available:
            meta.img_paths = meta.video_set
            meta.xyz = np.zeros((12, 3))
            meta.uv = np.zeros((len(meta.cam_range), 12, 2))
            meta.uv_merged = meta.uv
            meta.vis = np.zeros((12, ))
            meta.vis_merged = np.zeros((len(meta.cam_range), 12))

        else:
            raise NotImplementedError()


    return metas


def augment_hue(meta):
    tmp = list()
    for img in meta.img_list:
        hue_delta_deg = random.uniform(-10.0, 10.0)
        assert img.dtype == np.float32, 'Assumed datatype mismatch.'
        hsv = cv2.cvtColor(img + 0.5, cv2.COLOR_BGR2HSV)
        h = hsv[:, :, 0]
        h += hue_delta_deg

        # correct periodicity
        m = h > 360.0
        h[m] -= 360.0
        m = h < 0.0
        h[m] += 360.0

        # write result back
        hsv[:, :, 0] = h
        img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR) - 0.5
        tmp.append(img)
    meta.img_list = tmp
    return meta


def augment_blur(meta):
    tmp = list()
    for img in meta.img_list:
        assert img.dtype == np.float32, 'Assumed datatype mismatch.'

        sigma = np.random.rand() * 0.7
        tmp.append(cv2.GaussianBlur(img, (7, 7), sigma))
    meta.img_list = tmp
    return meta


def augment_snp_noise(meta):
    tmp = list()
    for img in meta.img_list:
        prob2flip = 0.001
        flip2zero = np.random.rand(*img.shape) < prob2flip
        flip2one = np.random.rand(*img.shape) < prob2flip
        img[flip2zero] = 0.0
        img[flip2one] = 255.0
        tmp.append(img)
    meta.img_list = tmp
    return meta


def augment_root(meta):
    noise = np.random.randn(3) * 0.0025  # 2.5 mm var
    meta.voxel_root = meta.voxel_root + noise
    return meta


def _vox2metric(coords_vox, root_metric):
    """ Converts voxel coordinates in metric coordinates. """
    global g_voxel_dim, g_voxel_resolution
    coords_metric = g_voxel_resolution / (g_voxel_dim-1) * (coords_vox - (g_voxel_dim-1) / 2.0) + root_metric
    return coords_metric


def _metric2vox(coords_metric, root_metric):
    """ Converts voxel coordinates in metric coordinates. """
    global g_voxel_dim, g_voxel_resolution
    coords_vox = (coords_metric - root_metric) * (g_voxel_dim-1) / g_voxel_resolution + (g_voxel_dim-1) / 2.0
    return coords_vox


def pose_to_img(meta):
    if meta is None:
        return None

    # calculate voxel coordinates
    xyz_vox = _metric2vox(meta[0].xyz, meta[0].voxel_root)

    return [
        np.expand_dims(meta[0].is_supervised, 0),
        np.expand_dims(meta[0].img_paths, 0),
        np.stack(meta[0].img_list) / 255.0 - 0.5,
        np.stack(meta[0].M_list),
        np.stack(meta[0].K_list),
        np.expand_dims(meta[0].xyz, 0),
        np.expand_dims(xyz_vox, 0),
        np.stack(meta[0].uv),
        np.expand_dims(meta[0].vis, 0),
        np.stack(meta[0].uv_merged),
        meta[0].vis_merged,
        np.reshape(meta[0].voxel_root, [1, 1, 3])
    ]


def df2dict(dp):
    keys = [data_t.is_supervised,
            data_t.ident1,
            data_t.image,
            data_t.M,
            data_t.K,
            data_t.xyz_nobatch,
            data_t.xyz_vox_nobatch,
            data_t.uv,
            data_t.vis_nobatch,
            data_t.uv_merged,
            data_t.vis_merged,
            data_t.voxel_root]

    values = dict()
    for k, v in zip(keys, dp):
        values[k] = v
    return values

