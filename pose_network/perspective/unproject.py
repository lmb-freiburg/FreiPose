from __future__ import unicode_literals, print_function
import tensorflow as tf


def _get_voxel_grid_points(voxel_grid_root, dimension=64, resolution=1.0):
    """ Calculates the world points surrounding the root point. """
    with tf.variable_scope('VoxelGridPoints'):
        one_vox = tf.constant(resolution / float(dimension-1))  # step between two voxels

        # coordinates of the voxelspace
        range = tf.range(0, dimension)
        X, Y, Z = tf.meshgrid(range, range, range, indexing='ij')
        X, Y, Z = tf.reshape(X, [-1]), tf.reshape(Y, [-1]), tf.reshape(Z, [-1])
        coords_vox = tf.expand_dims(tf.stack([X, Y, Z], 1), 0)

        # turn voxel space into real coordinates
        X = (tf.cast(X, tf.float32) - (dimension-1) / 2.0) * one_vox
        Y = (tf.cast(Y, tf.float32) - (dimension-1) / 2.0) * one_vox
        Z = (tf.cast(Z, tf.float32) - (dimension-1) / 2.0) * one_vox

        # make batch dependent
        s = voxel_grid_root.get_shape().as_list()
        X = tf.tile(tf.reshape(X, [1, -1]), [s[0], 1])
        Y = tf.tile(tf.reshape(Y, [1, -1]), [s[0], 1])
        Z = tf.tile(tf.reshape(Z, [1, -1]), [s[0], 1])

        # turn into coords
        coords_xyz = tf.stack([X, Y, Z], 2) + voxel_grid_root
        return coords_vox, coords_xyz


def _repeat(x, n_repeats):
    with tf.variable_scope('_repeat'):
        rep = tf.transpose(
            tf.expand_dims(tf.ones(shape=tf.stack([n_repeats, ])), 1), [1, 0])
        rep = tf.cast(rep, 'int32')
        x = tf.matmul(tf.reshape(x, (-1, 1)), rep)
    return tf.reshape(x, [-1])


def _bilinear_interpolate(im, x, y, out_size):
    with tf.variable_scope('_bilinear_interpolate'):
        # constants
        num_batch = tf.shape(im)[0]
        height = tf.shape(im)[1]
        width = tf.shape(im)[2]
        channels = tf.shape(im)[3]

        x = tf.cast(x, 'float32')
        y = tf.cast(y, 'float32')
        height_f = tf.cast(height, 'float32')
        width_f = tf.cast(width, 'float32')
        out_height = out_size[0]
        out_width = out_size[1]
        zero = tf.zeros([], dtype='int32')
        max_y = tf.cast(tf.shape(im)[1] - 1, 'int32')
        max_x = tf.cast(tf.shape(im)[2] - 1, 'int32')

        # # scale indices from [-1, 1] to [0, width/height]
        # x = (x + 1.0) * (width_f) / 2.0
        # y = (y + 1.0) * (height_f) / 2.0

        # do sampling
        x0 = tf.cast(tf.floor(x), 'int32')
        x1 = x0 + 1
        y0 = tf.cast(tf.floor(y), 'int32')
        y1 = y0 + 1

        x0 = tf.clip_by_value(x0, zero, max_x)
        x1 = tf.clip_by_value(x1, zero, max_x)
        y0 = tf.clip_by_value(y0, zero, max_y)
        y1 = tf.clip_by_value(y1, zero, max_y)
        dim2 = width
        dim1 = width * height
        base = _repeat(tf.range(num_batch) * dim1, out_height * out_width)
        base_y0 = base + y0 * dim2
        base_y1 = base + y1 * dim2
        idx_a = base_y0 + x0
        idx_b = base_y1 + x0
        idx_c = base_y0 + x1
        idx_d = base_y1 + x1

        # use indices to lookup pixels in the flat image and restore
        # channels dim
        im_flat = tf.reshape(im, tf.stack([-1, channels]))
        im_flat = tf.cast(im_flat, 'float32')
        Ia = tf.gather(im_flat, idx_a)
        Ib = tf.gather(im_flat, idx_b)
        Ic = tf.gather(im_flat, idx_c)
        Id = tf.gather(im_flat, idx_d)

        # and finally calculate interpolated values
        x0_f = tf.cast(x0, 'float32')
        x1_f = tf.cast(x1, 'float32')
        y0_f = tf.cast(y0, 'float32')
        y1_f = tf.cast(y1, 'float32')
        wa = tf.expand_dims(((x1_f - x) * (y1_f - y)), 1)
        wb = tf.expand_dims(((x1_f - x) * (y - y0_f)), 1)
        wc = tf.expand_dims(((x - x0_f) * (y1_f - y)), 1)
        wd = tf.expand_dims(((x - x0_f) * (y - y0_f)), 1)
        output = tf.add_n([wa * Ia, wb * Ib, wc * Ic, wd * Id])
        return output


def unproject(tensor,
              cam_intrinsic, cam_extrinsic,
              voxel_root_xyz_w, voxel_dim=64, resolution=2.0,
              return_image_coords=False):
    """ Unprojects an image-style tensor into a voxel grid.
        tensor: Tensor of shape [B, H, W, C]
        cam_intrinsic: bx3x3 intrinsic camera matrix: K
        cam_extrinsic: bx4x4 extrinsic camera matrix: M, representing the transformation from world to camera
        voxel_root_xyz_w: Tensor of shape [1, 3] that indicates the center of the voxel cube in world coords.
        voxel_dim: Grid resolution of the cube.
        resolution: scalar describing world edge length of the voxel.
    """
    with tf.variable_scope('unproject'):
        s = tensor.get_shape().as_list()
        num_batch = s[0]
        height = s[1]
        width = s[2]
        num_chan = s[3]

        # projection matrix
        cam_projection = tf.matmul(cam_intrinsic, cam_extrinsic[:, :3, :])

        # get voxel center coordinates
        coords_vox, coords_xyz = _get_voxel_grid_points(voxel_root_xyz_w, dimension=voxel_dim, resolution=resolution)

        # project voxelgrid into view
        coords_xyz_h = tf.concat([coords_xyz, tf.ones_like(coords_xyz[:, :, :1])], 2)
        coord_uv = tf.matmul(coords_xyz_h, tf.transpose(cam_projection, [0, 2, 1]))
        coord_uv = coord_uv[:, :, :2] / coord_uv[:, :, -1:]

        # bilinear resampling
        tensor_samples = tf.contrib.resampler.resampler(tensor, coord_uv) # this guy needs uv coordinates... didnt see that coming

        # put sampled values into the voxel
        num_batch, num_points, num_chan = tensor_samples.get_shape().as_list()
        tensor_samples = tf.reshape(tf.transpose(tensor_samples, [1, 0, 2]), [num_points, -1])  #flatten batch + channel into single dim
        coords_vox = tf.stop_gradient(tf.squeeze(coords_vox, 0)) # get rid of meaningless batch dimension (its the same grid across all the batch)
        voxel = tf.scatter_nd(coords_vox, tensor_samples, (voxel_dim, voxel_dim, voxel_dim, num_batch*num_chan))
        voxel = tf.transpose(tf.reshape(voxel, [voxel_dim, voxel_dim, voxel_dim, num_batch, num_chan]), [3, 0, 1, 2, 4])

        if return_image_coords:
            return voxel, coords_xyz, coord_uv
        return voxel, coords_xyz

totalMem = 0
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
    global totalMem
    totalMem += memory
    print(name, 'consumes ', memory/1024.0/1024.0, 'MB')
    print('total consumption ', totalMem/1024.0/1024.0, 'MB\n')
    return