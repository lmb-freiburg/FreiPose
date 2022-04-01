""" Yet another wrapper around a wrapper ..."""
from __future__ import print_function, unicode_literals
import numpy as np

import utils.CamLib as cl

G_TRIANG_TOOL_AVAIL = True
try:
    from .triangulate.TriangTool import TriangTool, t_triangulation

except Exception as e:
    print('Import of triangulation library failed. Maybe you still need to build it for your machine?\n'
          'cd utils/triangulate'
          'python setup.py build_ext --inplace')
    G_TRIANG_TOOL_AVAIL = False
    raise Exception


G_TRIANG_TOOL = None
def triangulate_robust(kp_uv, vis, K_list, M_list, dist_list=None,
                       threshold=50.0, mode=None):
    """ Given some 2D observations kp_uv and their respective validity this function finds 3D point hypothesis
        kp_uv: NxKx2 2D points
        vis: NxK visibility
        K_list Nx3x3 camera intrinsic
        dist_list Nx1x5 camera distortion
        M_list Nx4x4 camera extrinsic

        points3d: Kx3 3D point hypothesis
        points2d_proj: NxKx2 projection of points3d into the N cameras
        vis3d: K Validity of points3d
        points2d_merged: NxKx2 Merged result of input observations and points2d_proj according to
                points2d_merged = points2d_proj if ~vis else kp_uv
            So it basically keeps the 2D annotations and uses the reprojection if there was no annotation.
        vis2d_merged: NxK Validity of points2d_merged.
    """
    global G_TRIANG_TOOL, G_TRIANG_TOOL_AVAIL
    if not G_TRIANG_TOOL_AVAIL:
        print('THIS IS THE PROBLEM')
        raise ImportError('Could not load the triangulation toolbox.')

    # create tool if necessary
    if G_TRIANG_TOOL is None:
        G_TRIANG_TOOL = TriangTool()

    if mode is None:
        mode = t_triangulation.RANSAC

    # output values
    num_cams, num_kp = kp_uv.shape[:2]
    points3d = np.zeros((num_kp, 3), dtype=np.float32)
    vis3d = np.zeros((num_kp,), dtype=np.float32)
    points2d_proj = np.zeros((num_cams, num_kp, 2), dtype=np.float32)

    points2d_merged = kp_uv.copy()  # merged result of projection and 2d annotation (uses 2d anno if avail, otherwise 3d proj)
    vis2d_merged = vis.copy()  # validity of points2d_merged

    # iterate over keypoints
    for kp_id in range(num_kp):
        # resort data
        points2d = list()
        cams = list()
        for cid in range(num_cams):
            if vis[cid, kp_id] > 0.5:
                points2d.append(kp_uv[cid, kp_id])
                cams.append(cid)

        if np.unique(cams).shape[0] >= 2:
            # find consistent 3D hypothesis for the center of bounding boxed
            point3d, inlier = G_TRIANG_TOOL.triangulate([K_list[i] for i in cams],
                                                      [M_list[i] for i in cams],
                                                      np.expand_dims(np.array(points2d), 1),
                                                        dist_list=dist_list,
                                                      mode=mode,
                                                      threshold=threshold)
            if np.sum(inlier) >= 2:
                points3d[kp_id] = point3d
                vis3d[kp_id] = 1.0
                for cid, (K, M) in enumerate(zip(K_list, M_list)):
                    points2d_proj[cid, kp_id] = cl.project(cl.trafo_coords(point3d, M), K)

                    is_outlier_label = np.linalg.norm(points2d_proj[cid, kp_id] - kp_uv[cid, kp_id]) >= 2*threshold

                    # fill in projection into merged, if this point was not visible before
                    if vis2d_merged[cid, kp_id] < 0.5:
                        # if the 2D point was not filled before set projection
                        points2d_merged[cid, kp_id] = points2d_proj[cid, kp_id]
                        vis2d_merged[cid, kp_id] = 1.0

                    elif is_outlier_label:
                        # some 2D labels are just weird outliers:
                        # If the distance between a consistent 3D points proj and the label is too big we stick with the 3D pt
                        points2d_merged[cid, kp_id] = points2d_proj[cid, kp_id]
                        vis2d_merged[cid, kp_id] = 1.0

    return points3d, points2d_proj, vis3d, points2d_merged, vis2d_merged
