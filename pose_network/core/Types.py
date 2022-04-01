from __future__ import print_function, unicode_literals
from collections import namedtuple


Task = namedtuple('Task', ['name', 'freq', 'offset', 'pre_func', 'post_func'])
Loss = namedtuple('Loss', ['target', 'type', 'flag', 'weight', 'gt', 'pred'])
Summary = namedtuple('Summary', ['type', 'inputs', 'sum'])


# Summary types
class summary_t:
    always = 'always'
    sometimes = 'sometimes'


# Loss/Optimizer target
class target_t:
    generator = 'generator'
    discriminator = 'discriminator'

class eval_t:
    pose2d3d = 'pose2d3d'

# Loss types
class loss_t:
    l2 = 'l2'
    l1 = 'l1'
    l2_scorevolume = 'l2_scorevolume'
    l2_scoremap = 'l2_scoremap'
    limb_length = 'limb_length'
    limb_angles = 'limb_angles'


# Data types
class data_t:
    is_supervised = 'is_supervised'
    ident1 = 'ident1'

    image = 'image'
    K = 'K'
    M = 'M'
    voxel_root = 'voxel_root'
    xyz_nobatch = 'xyz_nobatch'
    xyz_vox_nobatch = 'xyz_vox_nobatch'

    uv = 'uv'
    uv_merged = 'uv_merged'
    vis_merged = 'vis_merged'
    vis_nobatch = 'vis_nobatch'
    scoremap = 'scoremap'

    pred_uv = 'pred_uv'
    pred_score2d = 'pred_score2d'
    pred_xyz = 'pred_xyz'
    pred_score3d = 'pred_score3d'

    pred_scoremap = 'pred_scoremap'
    pred_scorevol = 'pred_scorevol'

    pred_xyz_refine = 'pred_xyz_refine'
    pred_vis3d_refine = 'pred_vis3d_refine'
    pred_uv_refine = 'pred_uv_refine'

    pred_xyz_final = 'pred_xyz_final'
    pred_uv_final = 'pred_uv_final'



# Dataflow
class dataflow_t:
    rat_pose_mv = 'rat_pose_mv'
    rat_pose_mv_semi = 'rat_pose_mv_semi'


# Networks
class network_t:
    RatNetMV = 'RatNetMV'
