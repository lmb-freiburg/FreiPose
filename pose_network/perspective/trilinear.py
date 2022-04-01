import tensorflow as tf


def _repeat(x, n_repeats):
    with tf.variable_scope('_repeat'):
        rep = tf.transpose(
            tf.expand_dims(tf.ones(shape=tf.stack([
                n_repeats,
            ])), 1), [1, 0])
        rep = tf.to_int32(rep)
        x = tf.matmul(tf.reshape(x, (-1, 1)), rep)
        return tf.reshape(x, [-1])


def interpolate(im, x, y, z, out_size):
    """Trilinear interpolation layer.

    Args:
      im: A 5D tensor of size [num_batch, depth(X), height(Y), width(Z), num_channels].
        It is the input volume for the transformation layer (tf.float32).
      x: A tensor of size [num_batch, out_depth, out_height, out_width]
        representing the inverse coordinate mapping for x (tf.float32).
      y: A tensor of size [num_batch, out_depth, out_height, out_width]
        representing the inverse coordinate mapping for y (tf.float32).
      z: A tensor of size [num_batch, out_depth, out_height, out_width]
        representing the inverse coordinate mapping for z (tf.float32).
      out_size: A tuple representing the output size of transformation layer
        (float).

    Returns:
      A transformed tensor (tf.float32) of size [num_batch, out_size[0], out_size[1], out_size[2], num_channels]

    Notes:
      x, y, z are indices that correspond to the dimensions in 'im':
       X = depth
       Y = height
       Z = width
      The voxel tensor 'im' implicitly defines a coordinate system through its dimensions and x, y, z respresent
      indices in this coordinates. This function calculates their value by interpolating the neighboring values
      and reshape the result to desired out_size, which is a reasonable thing to do because the points in x, y, z
      define such a grid (they were created with meshgrid + trafo).

      rng = tf.arange(vox_dim)
      X, Y, Z = tf.meshgrid(rng, rng, rgn)
      T_ip = interpolate(T, X, Y, Z, vox_dim)

      T != T because at the boundaries values are missing for interpolation.

    """
    with tf.variable_scope('interpolate'):
        num_batch = im.get_shape().as_list()[0]
        depth = im.get_shape().as_list()[1]
        height = im.get_shape().as_list()[2]
        width = im.get_shape().as_list()[3]
        channels = im.get_shape().as_list()[4]

        x = tf.to_float(x)
        y = tf.to_float(y)
        z = tf.to_float(z)

        # Number of disparity interpolated.
        out_depth = out_size[0]
        out_height = out_size[1]
        out_width = out_size[2]
        zero = tf.zeros([], dtype='int32')
        # 0 <= z < depth, 0 <= y < height & 0 <= x < width.
        max_z = tf.to_int32(tf.shape(im)[1] - 1)
        max_y = tf.to_int32(tf.shape(im)[2] - 1)
        max_x = tf.to_int32(tf.shape(im)[3] - 1)

        x0 = tf.to_int32(tf.floor(x))
        x1 = x0 + 1
        y0 = tf.to_int32(tf.floor(y))
        y1 = y0 + 1
        z0 = tf.to_int32(tf.floor(z))
        z1 = z0 + 1

        x0_clip = tf.clip_by_value(x0, zero, max_x)
        x1_clip = tf.clip_by_value(x1, zero, max_x)
        y0_clip = tf.clip_by_value(y0, zero, max_y)
        y1_clip = tf.clip_by_value(y1, zero, max_y)
        z0_clip = tf.clip_by_value(z0, zero, max_z)
        z1_clip = tf.clip_by_value(z1, zero, max_z)
        dim3 = width
        dim2 = width * height
        dim1 = width * height * depth
        base = _repeat(
            tf.range(num_batch) * dim1, out_depth * out_height * out_width)
        base_z0_y0 = base + z0_clip * dim2 + y0_clip * dim3
        base_z0_y1 = base + z0_clip * dim2 + y1_clip * dim3
        base_z1_y0 = base + z1_clip * dim2 + y0_clip * dim3
        base_z1_y1 = base + z1_clip * dim2 + y1_clip * dim3

        idx_z0_y0_x0 = base_z0_y0 + x0_clip
        idx_z0_y0_x1 = base_z0_y0 + x1_clip
        idx_z0_y1_x0 = base_z0_y1 + x0_clip
        idx_z0_y1_x1 = base_z0_y1 + x1_clip
        idx_z1_y0_x0 = base_z1_y0 + x0_clip
        idx_z1_y0_x1 = base_z1_y0 + x1_clip
        idx_z1_y1_x0 = base_z1_y1 + x0_clip
        idx_z1_y1_x1 = base_z1_y1 + x1_clip

        # Use indices to lookup pixels in the flat image and restore
        # channels dim
        im_flat = tf.reshape(im, tf.stack([-1, channels]))
        im_flat = tf.to_float(im_flat)
        i_z0_y0_x0 = tf.gather(im_flat, idx_z0_y0_x0)
        i_z0_y0_x1 = tf.gather(im_flat, idx_z0_y0_x1)
        i_z0_y1_x0 = tf.gather(im_flat, idx_z0_y1_x0)
        i_z0_y1_x1 = tf.gather(im_flat, idx_z0_y1_x1)
        i_z1_y0_x0 = tf.gather(im_flat, idx_z1_y0_x0)
        i_z1_y0_x1 = tf.gather(im_flat, idx_z1_y0_x1)
        i_z1_y1_x0 = tf.gather(im_flat, idx_z1_y1_x0)
        i_z1_y1_x1 = tf.gather(im_flat, idx_z1_y1_x1)

        # Finally calculate interpolated values.
        x0_f = tf.to_float(x0)
        x1_f = tf.to_float(x1)
        y0_f = tf.to_float(y0)
        y1_f = tf.to_float(y1)
        z0_f = tf.to_float(z0)
        z1_f = tf.to_float(z1)
        # Check the out-of-boundary case.
        x0_valid = tf.to_float(
            tf.less_equal(x0, max_x) & tf.greater_equal(x0, 0))
        x1_valid = tf.to_float(
            tf.less_equal(x1, max_x) & tf.greater_equal(x1, 0))
        y0_valid = tf.to_float(
            tf.less_equal(y0, max_y) & tf.greater_equal(y0, 0))
        y1_valid = tf.to_float(
            tf.less_equal(y1, max_y) & tf.greater_equal(y1, 0))
        z0_valid = tf.to_float(
            tf.less_equal(z0, max_z) & tf.greater_equal(z0, 0))
        z1_valid = tf.to_float(
            tf.less_equal(z1, max_z) & tf.greater_equal(z1, 0))

        w_z0_y0_x0 = tf.expand_dims(((x1_f - x) * (y1_f - y) *
                                     (z1_f - z) * x1_valid * y1_valid * z1_valid),
                                    1)
        w_z0_y0_x1 = tf.expand_dims(((x - x0_f) * (y1_f - y) *
                                     (z1_f - z) * x0_valid * y1_valid * z1_valid),
                                    1)
        w_z0_y1_x0 = tf.expand_dims(((x1_f - x) * (y - y0_f) *
                                     (z1_f - z) * x1_valid * y0_valid * z1_valid),
                                    1)
        w_z0_y1_x1 = tf.expand_dims(((x - x0_f) * (y - y0_f) *
                                     (z1_f - z) * x0_valid * y0_valid * z1_valid),
                                    1)
        w_z1_y0_x0 = tf.expand_dims(((x1_f - x) * (y1_f - y) *
                                     (z - z0_f) * x1_valid * y1_valid * z0_valid),
                                    1)
        w_z1_y0_x1 = tf.expand_dims(((x - x0_f) * (y1_f - y) *
                                     (z - z0_f) * x0_valid * y1_valid * z0_valid),
                                    1)
        w_z1_y1_x0 = tf.expand_dims(((x1_f - x) * (y - y0_f) *
                                     (z - z0_f) * x1_valid * y0_valid * z0_valid),
                                    1)
        w_z1_y1_x1 = tf.expand_dims(((x - x0_f) * (y - y0_f) *
                                     (z - z0_f) * x0_valid * y0_valid * z0_valid),
                                    1)

        output = tf.add_n([
            w_z0_y0_x0 * i_z0_y0_x0, w_z0_y0_x1 * i_z0_y0_x1,
            w_z0_y1_x0 * i_z0_y1_x0, w_z0_y1_x1 * i_z0_y1_x1,
            w_z1_y0_x0 * i_z1_y0_x0, w_z1_y0_x1 * i_z1_y0_x1,
            w_z1_y1_x0 * i_z1_y1_x0, w_z1_y1_x1 * i_z1_y1_x1
        ])
        return tf.reshape(output, [num_batch, out_size[0], out_size[1], out_size[2], channels])


if __name__ == '__main__':
    def dump_origin(file_out, offset=None):
        """ Write camera calibration to disk for debugging. """
        if offset is None:
            offset = np.array([[0.0, 0.0, 0.0]])

        vertices = np.array([[0.0, 0.0, 0.0],
                             [1.0, 0.0, 0.0],
                             [0.0, 1.0, 0.0],
                             [0.0, 0.0, 1.0]])
        vertices += offset

        edges = [[0, 1],
                 [0, 2],
                 [0, 3]]

        color = [[255, 255, 255],
                 [255, 0, 0],
                 [0, 255, 0],
                 [0, 0, 255]]

        vertices = np.array(vertices).astype(np.float32)
        edges = np.array(edges).astype(np.int32)
        color = np.array(color).astype(np.uint8)

        this_file_out = file_out + '.ply'
        with open(this_file_out, 'w') as fo:
            # header
            fo.write('ply\n')
            fo.write('format ascii 1.0\n')

            fo.write('element vertex %d\n' % len(vertices))
            fo.write('property float x\n')
            fo.write('property float y\n')
            fo.write('property float z\n')
            fo.write('property uchar red                   { start of vertex color }\n')
            fo.write('property uchar green\n')
            fo.write('property uchar blue\n')

            fo.write('element edge %d\n' % len(edges))
            fo.write('property int vertex1\n')
            fo.write('property int vertex2\n')
            fo.write('property uchar red                   { start of edge color }\n')
            fo.write('property uchar green\n')
            fo.write('property uchar blue\n')

            fo.write('end_header\n')

            # data
            for v, c in zip(vertices, color):
                fo.write('%f %f %f %d %d %d\n' % (v[0], v[1], v[2],
                                                  c[0], c[1], c[2]))

            for c, (i, j) in zip(color[1:], edges):
                fo.write('%d %d %d %d %d\n' % (i, j, c[0], c[1], c[2]))
            print('Saved ply to: %s' % this_file_out)


    def dump_point_array_ply(file_out, points, conf):
        """ Write a list of points with confidences to disk for debugging. """
        import matplotlib.pyplot as plt
        heat_map = plt.get_cmap('hot')
        conf = (conf - conf.min()) / (conf.max() - conf.min())
        colors = heat_map(conf.flatten()) * 255

        with open(file_out + '.ply', 'w') as fo:
            # header
            fo.write('ply\n')
            fo.write('format ascii 1.0\n')

            fo.write('element vertex %d\n' % points.shape[0])
            fo.write('property float x\n')
            fo.write('property float y\n')
            fo.write('property float z\n')
            fo.write('property uchar red                   { start of vertex color }\n')
            fo.write('property uchar green\n')
            fo.write('property uchar blue\n')

            fo.write('end_header\n')

            # data
            for c, v in zip(colors, points):
                fo.write('%f %f %f %d %d %d\n' % (v[0], v[1], v[2], c[0], c[1], c[2]))

        print('Saved ply to: %s' % file_out)

    import numpy as np
    vox_dim = 4
    voxel = np.zeros((1, vox_dim, vox_dim, vox_dim, 1))
    # voxel[:] = 1.0
    # voxel[0, 0, 0, :, 0] = 1.0
    # voxel[0, 0, :, 0, 0] = 1.0
    voxel[0, 1, 1, 1, 0] = 1.0

    dump_origin('/home/egg/ply_dump/origin')

    print('voxel DEPTH=0', voxel[0, 0, :, :, 0])
    print('voxel DEPTH=1', voxel[0, 1, :, :, 0])
    print('voxel DEPTH=2', voxel[0, 2, :, :, 0])
    print('voxel DEPTH=4', voxel[0, 3, :, :, 0])

    # this is the grid voxel defines (by its dimensions)
    DEPTH, HEIGHT, WIDTH = np.meshgrid(np.arange(vox_dim), np.arange(vox_dim), np.arange(vox_dim), indexing='ij')
    DEPTH, HEIGHT, WIDTH = np.reshape(DEPTH, [-1]), np.reshape(HEIGHT, [-1]), np.reshape(WIDTH, [-1])
    DEPTH, HEIGHT, WIDTH = DEPTH.astype(np.float32), HEIGHT.astype(np.float32), WIDTH.astype(np.float32)
    coords = np.stack([DEPTH, HEIGHT, WIDTH], 1)

    dump_point_array_ply('/home/egg/ply_dump/vox_in', coords, np.reshape(voxel, [-1]))

    # modify the grid (this yields new points in voxel's implicit coordinate system)
    # WIDTH = WIDTH-0.5 # translate
    # HEIGHT = HEIGHT-1
    # DEPTH = DEPTH-0.5
    # DEPTH = DEPTH-1

    # mirror grid along width dimension
    # WIDTH = vox_dim - WIDTH - 1
    # HEIGHT = vox_dim - HEIGHT - 1
    # DEPTH = vox_dim - DEPTH - 1

    # # rotate with matrix
    # print('coords', coords[:5, :])
    c = (vox_dim-1)/2.0
    print('vox center', c)
    coords -= (vox_dim-1)/2.0
    dump_origin('/home/egg/ply_dump/origin_t', np.zeros((1, 3))+(vox_dim-1)/2.0)
    import cv2
    deg = 20
    rad = deg/180.0*np.pi
    R, _ = cv2.Rodrigues(np.array([0.0, rad, 0.0]))
    # R, _ = cv2.Rodrigues(np.array([0.0, 0.0, np.pi]))
    # R = np.eye(3)
    coords_t = np.matmul(coords, R.T)
    coords_t += (vox_dim-1)/2.0
    # print('coords_t', coords_t[:5, :])
    DEPTH, HEIGHT, WIDTH = coords_t[:, 0], coords_t[:, 1], coords_t[:, 2]

    dump_point_array_ply('/home/egg/ply_dump/vox_t_%d' % deg, coords_t, np.reshape(voxel, [-1]))

    # calculate values at the new point locations
    voxel_tf = tf.constant(voxel)
    print('voxel_tf', voxel_tf)
    print('DEPTH', DEPTH.shape)
    print('HEIGHT', HEIGHT.shape)
    print('WIDTH', WIDTH.shape)
    print('vox_dim', vox_dim.shape)
    voxel_ip_tf = interpolate(voxel_tf, DEPTH, HEIGHT, WIDTH, [vox_dim]*3)

    session = tf.Session()
    voxel_ip = session.run(voxel_ip_tf)

    print('voxel_ip DEPTH=0', voxel_ip[0, 0, :, :, 0])
    print('voxel_ip DEPTH=1', voxel_ip[0, 1, :, :, 0])
    print('voxel_ip DEPTH=2', voxel_ip[0, 2, :, :, 0])
    print('voxel_ip DEPTH=4', voxel_ip[0, 3, :, :, 0])

