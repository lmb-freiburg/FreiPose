from __future__ import unicode_literals, print_function
import tensorflow as tf
import numpy as np

from trilinear import interpolate


def _tf_repeat(tensor, repetitions):
    """ Mimics behavior of np.repeat for tensorflow. """
    with tf.name_scope('repeat'):
        tensor = tf.reshape(tensor, [-1, 1])  # Convert to a len(yp) x 1 matrix.
        tensor = tf.tile(tensor, [1, repetitions])  # Create multiple columns.
        return tf.reshape(tensor, [-1])  # Convert back to a vector.


def get_cam_frustum_points(cam_intrinsic_norm, voxel_root_xyz_cam,
                           voxel_dim=64, resolution=1.0):
    """ Calculates world points that form the cameras frustum.

        Basically gives you a (not equally spaced) volume where each line along the z direction corresponds to an camera ray.

        cam_intrinsic_norm: tensor of shape [B, 3, 3] representing the normalized intrinsic matrix (normalized by image dimensions.)
        voxel_root_xyz_cam: tensor of shape [B, 3] representing the voxel center in camera 3d coordinates (voxel_dim//2, voxel_dim//2, voxel_dim//2).
        resolution: scalar describing world edge length of the voxel.
    """
    with tf.variable_scope('CamFrustrum'):
        batch_num = cam_intrinsic_norm.get_shape().as_list()[0]

        # coordinates of image space
        range = tf.linspace(0.0, 1.0, voxel_dim+1)[:-1]  # normalized image coordinates
        H, W = tf.meshgrid(range, range, indexing='ij')  # get sample point in the image plane
        H, W = tf.reshape(H, [-1]), tf.reshape(W, [-1])
        coords_hw = tf.expand_dims(tf.stack([H, W], 1), 0)  # this is [1, voxel_dim^2, 2]

        # calculate rays
        coords_hw_hom = tf.concat([coords_hw, tf.ones_like(coords_hw[:, :, :1])], 2) # this is [1, voxel_dim^2, 3]
        coords_hw_hom = tf.tile(coords_hw_hom, [batch_num, 1, 1])  # this is [B, voxel_dim^2, 3]
        rays = tf.matmul(coords_hw_hom, tf.transpose(tf.matrix_inverse(cam_intrinsic_norm), [0, 2, 1]))  # this is [B, voxel_dim^2, 3]
        rays = tf.tile(rays, [1, voxel_dim, 1])  # this is [B, voxel_dim^3, 3]

        # space to get world points
        z_samples = tf.linspace(-1.0, 1.0, voxel_dim+1)[:-1]   # this is [voxel_dim]
        z_samples = tf.reshape(_tf_repeat(z_samples, voxel_dim*voxel_dim), [1, -1, 1])  # this is [1, voxel_dim^3, 1]
        z_center = voxel_root_xyz_cam[:, :, -1:] # this is [B, 1, 1]
        z_samples = z_samples*resolution/2.0 + z_center  # scale
        coords_xyz = tf.multiply(rays, z_samples)  # this is [B, voxel_dim^3, 3]
        return coords_xyz, coords_hw


def projection(voxel, voxel_root_xyz_w,
               cam_intrinsic_norm, cam_extrinsic,
               voxel_dim=64, resolution=1.0, name='Projection'):
    """ Projects an voxel-style tensor into a image style tensor.
        voxel: Tensor of shape [B, D, H, W, C]
        cam_intrinsic: bx3x3 intrinsic camera matrix: K
        cam_extrinsic: bx4x4 extrinsic camera matrix: M giving the transformation cam -> world
        voxel_root_xyz_w: Tensor of shape [1, 3] that indicates the center of the voxel cube in world coords.
        voxel_dim: Scalar grid resolution of the cube.
        resolution: scalar describing the world edge length of the voxel.
    """

    with tf.variable_scope(name):
        # get camera frustums (in 3D camera coordinates)
        voxel_root_xyz_w_hom = tf.concat([voxel_root_xyz_w, tf.ones_like(voxel_root_xyz_w[:, :, :1])], 2)
        voxel_root_xyz_cam = tf.matmul(voxel_root_xyz_w_hom, tf.transpose(tf.matrix_inverse(cam_extrinsic), [0, 2, 1]))
        voxel_root_xyz_cam = voxel_root_xyz_cam[:, :, :3] / voxel_root_xyz_cam[:, :, -1:]
        frustum_coords_xyz, coords_hw = get_cam_frustum_points(cam_intrinsic_norm, voxel_root_xyz_cam, voxel_dim, resolution)

        # transform it into world coordinates
        frustum_coords_xyz_hom = tf.concat([frustum_coords_xyz, tf.ones_like(frustum_coords_xyz[:, :, :1])], 2)
        frustum_coords_xyz_w = tf.matmul(frustum_coords_xyz_hom, tf.transpose(cam_extrinsic, [0, 2, 1]))
        frustum_coords_xyz_w = frustum_coords_xyz_w[:, :, :3] / frustum_coords_xyz_w[:, :, -1:]

        # transform into voxel index coordinates
        frustum_coords_vox = (frustum_coords_xyz_w - voxel_root_xyz_w) * (voxel_dim - 1) / resolution + voxel_dim // 2
        frustum_coords_vox = tf.stop_gradient(frustum_coords_vox)  # CONFIRM THIS: dont backprop through the indices, but only the values

        # interpolate values
        x = tf.reshape(frustum_coords_vox[:, :, 0], [-1])  #TODO: I'm a bit scared about this flattening, because
        y = tf.reshape(frustum_coords_vox[:, :, 1], [-1])  #TODO: it does not match with the description of interpolate
        z = tf.reshape(frustum_coords_vox[:, :, 2], [-1])  #TODO: But results seem to be okay
        voxel_frustum = interpolate(voxel, z, y, x, [voxel_dim] * 3)

        # project along z axis
        values_project = tf.reduce_max(voxel_frustum, 1)
        values_project = tf.transpose(values_project, [0, 2, 1, 3])

        return values_project, frustum_coords_xyz_w, coords_hw, frustum_coords_vox, frustum_coords_xyz, voxel_frustum


def _printMb(tensor, name=None):
    if name is None:
        name = tensor.name

    s = tensor.get_shape().as_list()
    if tensor.dtype == tf.float32:
        memory = 4
    elif tensor.dtype == tf.int32:
        memory = 4
    else:
        print('Unknown datatype.')

    for i in s:
        memory *= float(i)
    print(name, 'consumes ', memory/1024.0/1024.0, 'MB')
    return

