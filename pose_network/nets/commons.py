from __future__ import print_function, unicode_literals
import tensorflow as tf
import numpy as np


def trafoPoints(xyz, M):
    """ Transforms points into another coordinate frame. """
    with tf.name_scope('trafoPoints'):
        xyz_h = tf.concat([xyz, tf.ones_like(xyz[:, :, :1])], 2)
        xyz_cam = tf.matmul(xyz_h,
                                 tf.transpose(M, [0, 2, 1]))
        xyz_cam = xyz_cam[:, :, :3] / xyz_cam[:, :, -1:]
        return xyz_cam


def projectPoints(xyz, K):
    """ Project points into the camera. """
    with tf.name_scope('projectPoints'):
        uv = tf.matmul(xyz, tf.transpose(K, [0, 2, 1]))
        uv = uv[:, :, :2] / uv[:, :, -1:]
        return uv


def unproject(points2d, K, z=None, K_is_inv=False):
    """ Unproject a 2D point of camera K to distance z.
    """
    with tf.name_scope('unproject'):
        batch = K.get_shape().as_list()[0]
        points2d = tf.reshape(points2d, [batch, -1, 2])
        points2d_h = tf.concat([points2d, tf.ones_like(points2d[:, :, :1])], -1)  # homogeneous

        if K_is_inv:
            K_inv = K
        else:
            K_inv = tf.matrix_inverse(K)

        points3D = tf.matmul(points2d_h, tf.transpose(K_inv, [0, 2, 1]))  # 3d point corresponding to the estimate image point where the root should go to
        if z is not None:
            z = tf.reshape(z, [batch, -1, 1])
            points3D = points3D * z
    return points3D


def calc_eye_maps_approx(M, encoding):
    """ Calculates the vector where each pixel is looking at.
        K_tf: Bx3x3, camera intrinsics
        M_tf: Bx4x4, camera extrinsics, mapping world 2 camera.
        encoding: Just for the shape.
    """
    with tf.name_scope('calc_eye_maps_approx'):
        s = encoding.get_shape().as_list()  # B x N x N x C
        # 1. Approximate eye vector is last R column of M
        coords_xy = tf.reshape(M[:, :3, 2], [s[0], 1, 1, 3])
        eye_maps = tf.tile(coords_xy, [1, s[1], s[2], 1])
        return eye_maps


def calc_eye_maps(K_tf, M, encoding):
    """ Calculates the vector where each pixel is looking at.
        K_tf: Bx3x3, camera intrinsics
        M_tf: Bx4x4, camera extrinsics, mapping world 2 camera.
        encoding: Just for the shape.
    """
    with tf.name_scope('calc_eye_maps'):
        # 1. Get meshgrid
        s = encoding.get_shape().as_list()  # B x N x N x C
        rng_h = tf.range(s[1])
        rng_w = tf.range(s[2])
        H, W = tf.meshgrid(rng_h, rng_w, indexing='ij')
        H, W = tf.reshape(H, [-1]), tf.reshape(W, [-1])
        coords_uv = tf.expand_dims(tf.stack([W, H, tf.ones_like(H)], 1), 0)
        coords_uv = tf.tile(coords_uv, [s[0], 1, 1])
        coords_uv = tf.cast(coords_uv, tf.float32)  # B x N ^2 x 2

        # 2. Calculate eye vectors
        K_inv = tf.matrix_inverse(K_tf)
        coords_xy = tf.matmul(coords_uv, tf.transpose(K_inv, [0, 2, 1]))  # B x N ^2 x 3
        coords_xy /= tf.sqrt(tf.reduce_sum(tf.square(coords_xy), 2, keepdims=True))

        # 3. Trafo into world space
        coords_xy = tf.matmul(coords_xy, tf.transpose(M[:, :3, :3], [0, 2, 1]))

        # 3. Reshape to map
        eye_maps = tf.reshape(coords_xy, [s[0], s[1], s[2], 3])
        return eye_maps


def normalize_xyz(xyz):
    """ Normalizes coordinates by a reference length and a root keypoint. """
    with tf.name_scope('normalize_xyz'):
        length_ref = xyz[:, 9:10, :] - xyz[:, 10:11, :]
        length_ref = tf.sqrt(tf.reduce_sum(tf.square(length_ref), -1, keepdims=True))
        scale = 1.0 / length_ref

        # normalize 3D
        xyz = xyz*scale

        # get root relative depths (in the scale normed frame)
        root_norm_depth = xyz[:, 9:10, :]
        xyz = xyz - root_norm_depth

    return xyz, scale, root_norm_depth


def normalize_xyz_np(xyz):
    """ Normalizes coordinates by a reference length and a root keypoint. """
    length_ref = xyz[:, 9:10, :] - xyz[:, 10:11, :]
    length_ref = np.sqrt(np.sum(np.square(length_ref), -1, keepdims=True))
    scale = 1.0 / length_ref

    # normalize 3D
    xyz = xyz*scale

    # get root relative depths (in the scale normed frame)
    root_norm_depth = xyz[:, 9:10, :]
    xyz = xyz - root_norm_depth

    return xyz, scale, root_norm_depth


def denormalize_xyz(xyz, scale, root_norm_depth):
    """ Recovers full xyz coordinates using scale and global translation. """
    with tf.name_scope('normalize_xyz'):
        xyz = (xyz + root_norm_depth) / scale
    return xyz


def denormalize_xyz_np(xyz, scale, root_norm_depth):
    """ Recovers full xyz coordinates using scale and global translation. """
    xyz = (xyz + root_norm_depth) / scale
    return xyz


def _length(vec):
    return tf.sqrt(tf.reduce_sum(tf.square(vec), 1))


def _calc_hand_bone_lengths(xyz):
    kinematic = [(0, 1), (1, 2), (2, 3), (3, 4),
                 (0, 5), (5, 6), (6, 7), (7, 8),
                 (0, 9), (9, 10), (10, 11), (11, 12),
                 (0, 13), (13, 14), (14, 15), (15, 16),
                 (0, 17), (17, 18), (18, 19), (19, 20)]

    lengths = list()
    for pid, cid in kinematic:
        lengths.append(_length(xyz[:, pid, :] - xyz[:, cid, :]))
    lengths = tf.stack(lengths, 1)
    return lengths


def calc_hand_scale(xyz_gt, xyz_pred):
    with tf.name_scope('calc_hand_scale'):
        lengths_gt = _calc_hand_bone_lengths(xyz_gt)
        lengths_pred = _calc_hand_bone_lengths(xyz_pred)
        hand_scale = tf.reduce_mean(lengths_gt / lengths_pred, 1)
        hand_scale = tf.reshape(hand_scale, [xyz_gt.shape[0], 1, 1])
        return hand_scale


def get_global_rot_mat(xyz):
    """ Calculates the global rotation for a given hand. """
    # find new basis vectors
    v_x = xyz[:, :1, :] - xyz[:, 9:10, :]
    v_z = xyz[:, 5:6, :] - xyz[:, 9:10, :]
    v_y = tf.cross(v_z, v_x)
    v_z = tf.cross(v_x, v_y)
    R = tf.concat([v_x, v_y, v_z], 1)
    return R


def batch_inv_rodrigues(M, name=None):
    """
    From
    http://www.euclideanspace.com/maths/geometry/rotations/conversions/matrixToAngle/
    https://docs.opencv.org/2.4/modules/calib3d/doc/camera_calibration_and_3d_reconstruction.html#void%20Rodrigues(InputArray%20src,%20OutputArray%20dst,%20OutputArray%20jacobian)
    """
    with tf.name_scope(name, "batch_inv_rodrigues", [M]):
        r_x = M[:, 2, 1] - M[:, 1, 2]
        r_y = M[:, 0, 2] - M[:, 2, 0]
        r_z = M[:, 1, 0] - M[:, 0, 1]
        two_sin_theta = tf.sqrt(r_x*r_x + r_y*r_y + r_z*r_z + 1e-10)
        cos_theta = (M[:, 0, 0] + M[:, 1, 1] + M[:, 2, 2] - 1)/2
        theta = tf.acos(cos_theta)

        # Solution A: Works almost all the time
        r_x *= theta/two_sin_theta
        r_y *= theta/two_sin_theta
        r_z *= theta/two_sin_theta
        r = tf.stack([r_x, r_y, r_z], -1)

        # Solution B: for then theta is close to pi
        xx = (M[:, 0, 0]+1.0)/2.0
        yy = (M[:, 1, 1]+1.0)/2.0
        zz = (M[:, 2, 2]+1.0)/2.0
        xy = (M[:, 0, 1]+M[:, 1, 0])/4.0
        xz = (M[:, 0, 2]+M[:, 2, 0])/4.0
        yz = (M[:, 1, 2]+M[:, 1, 2])/4.0

        # Subcase: xx largest element
        r_x_1 = tf.sqrt(xx)
        r_y_1 = xy / xx
        r_z_1 = xz / xx
        r_1 = tf.stack([r_x_1, r_y_1, r_z_1], -1)
        r_1 *= theta

        # Subcase: yy largest element
        r_x_2 = xy / yy
        r_y_2 = tf.sqrt(yy)
        r_z_2 = xz / yy
        r_2 = tf.stack([r_x_2, r_y_2, r_z_2], -1)
        r_2 *= theta

        # Subcase: zz largest element
        r_x_3 = xz / zz
        r_y_3 = yz / zz
        r_z_3 = tf.sqrt(zz)
        r_3 = tf.stack([r_x_3, r_y_3, r_z_3], -1)
        r_3 *= theta

        # reduce subcases
        def _enlarge(x):
            return tf.tile(tf.expand_dims(x, -1), [1, 3])
        xx, yy, zz = _enlarge(xx), _enlarge(yy), _enlarge(zz)
        r_1 = tf.where(tf.logical_and(yy > xx, yy > zz), r_2, r_1)
        r_1 = tf.where(tf.logical_and(zz > xx, zz > yy), r_3, r_1)

        # select between A and B
        # problems occur at angles of 180deg, so sin ~ 0 and cos ~ +-1
        cond = tf.logical_and(tf.abs(two_sin_theta) < 1e-3, 1.0 - np.abs(cos_theta) < 1e-4)
        r = tf.where(_enlarge(cond), r_1, r)

        return r



def normalize_hand_pose(xyz):
    """ Given 3D hand pose coordinates this normalizes global translation, rotation and scale. """

    # xyz_n = xyz - xyz[:, :1, :]  # get rid of global translation
    xyz_n = xyz - tf.reduce_mean(xyz[:, :1, :], 0)  # get rid of global translation: make zero mean

    # normalize scale
    xyz_n /= tf.reshape(_length(xyz_n[:, 10, :] - xyz_n[:, 9, :]), [-1, 1, 1])

    # find global rotation
    R = get_global_rot_mat(xyz_n)

    xyz_n = tf.matmul(xyz_n, R)
    return xyz_n
