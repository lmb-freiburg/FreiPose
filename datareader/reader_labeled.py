from __future__ import print_function, unicode_literals
import tensorpack as tp
import random
import cv2
import sys
sys.path.append('/home/zimmermc/projects/ALTool')

from utils.general_util import compensate_crop_coords, compensate_crop_K
import utils.CamLib as cl
from pose_network.core.Types import *
from datareader.DatasetLabeled import *

from utils.general_util import json_load

g_voxel_dim, g_voxel_resolution = 64, 0.4


def build_dataflow(model, datasets_list,
                   is_train=False, threaded=False, shuffle=True,
                   single_sample=False,
                   voxel_dim=64, voxel_resolution=0.4):
    global g_voxel_dim, g_voxel_resolution
    g_voxel_dim, g_voxel_resolution = voxel_dim, voxel_resolution

    # build dataflow
    ds = DatasetLabeled(model, datasets_list, single_sample, shuffle=shuffle)
    if threaded:
        ds = tp.MultiThreadMapData(ds, map_func=read_img, nr_thread=8, buffer_size=16)
    else:
        ds = tp.MapData(ds, read_img)

    if is_train:
        ds = tp.MapDataComponent(ds, augment_hue)
        ds = tp.MapDataComponent(ds, augment_snp_noise)
        ds = tp.MapDataComponent(ds, augment_blur)
        ds = tp.MapDataComponent(ds, augment_warp)
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


def read_img(metas):
    for meta in metas:
        if not all([os.path.exists(x) for x in meta.img_paths]):
            print('Path not found: ', meta.img_paths[0])

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


def _compensate2D(uv, M_rot, pp):
    uv_centered = uv - np.reshape(pp, [1, 2])
    uv_centered = np.matmul(uv_centered, M_rot[:2, :2])
    return uv_centered + np.reshape(pp, [1, 2])

def augment_warp(meta):
    tmp_img, tmp_K = list(), list()
    for i in range(len(meta.K_list)):
        assert meta.img_list[i].dtype == np.float32, 'Assumed datatype mismatch.'

        ## ROTATE AROUND PP
        ang_range = 15.0
        angle = np.random.rand() * ang_range - ang_range/2.0
        pp = meta.K_list[i][:2, 2]
        M = cv2.getRotationMatrix2D((pp[0], pp[1]), angle=angle, scale=1.0)
        meta.img_list[i] = cv2.warpAffine(meta.img_list[i], M,
                                          meta.img_list[i].shape[:2][::-1], borderValue=(128, 128, 128))

        # compensate 3D
        angle *= np.pi/180.0
        c, s = np.cos(angle), np.sin(angle)
        M_rot = np.array([[c, -s, 0.0],
                         [s, c, 0.0],
                         [0.0, 0.0, 1.0]])
        M_rot4 = np.eye(4)
        M_rot4[:3, :3] = M_rot
        meta.M_list[i] = np.matmul(np.linalg.inv(M_rot4), meta.M_list[i])

        # compensate 2D
        meta.uv[i] = _compensate2D(meta.uv[i], M_rot, pp)
        meta.uv_merged[i] = _compensate2D(meta.uv_merged[i], M_rot, pp)

        ## SCALE AND CROP
        # sample warp params
        t_f = 5  # displacement in pix plus/minus
        s_f = 0.2
        s = np.random.rand() * s_f + 1.0  # scale, only 'zoom in'
        t = np.random.rand(2) * 2 * t_f - t_f  # translation

        # how much the image center is translated by scaling (compensate so the central pixel stays the same)
        off = 0.5 * (s - 1.0) * np.array(meta.img_list[i].shape[:2][::-1])

        M = np.array([[s, 0.0, t[0] - off[0]],
                      [0.0, s, t[1] - off[1]],
                      [0.0, 0.0, 1.0]])
        meta.img_list[i] = cv2.warpAffine(meta.img_list[i], M[:2, :], meta.img_list[i].shape[:2][::-1],
                                          flags=cv2.INTER_LINEAR, borderValue=(128, 128, 128))

        meta.K_list[i] = np.matmul(M, meta.K_list[i])
        meta.uv[i] = cl.trafo_coords(meta.uv[i], M)
        meta.uv_merged[i] = cl.trafo_coords(meta.uv_merged[i], M)
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
        1.0,
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

