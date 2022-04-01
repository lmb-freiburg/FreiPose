import tensorflow as tf
import numpy as np


def create_scoremaps_2d(kp_uv, sigma, img_shape_hw, vis=None):
    """ Creates scoremaps according to the given keypoint information as network ground truth.

    kp_uv: (K, 2)
    vis: (K, 1)
    sigma: scalar
    img_shape_hw: tuple containing height and width of the target image
    """
    if vis is None:
        vis = np.ones_like(kp_uv[:, :1])

    h_range = np.arange(0, img_shape_hw[0])
    w_range = np.arange(0, img_shape_hw[1])

    H, W = np.meshgrid(h_range, w_range, indexing='ij')
    H, W = H.flatten().astype(np.float32), W.flatten().astype(np.float32)
    coord_uv = np.expand_dims(np.stack([W, H], 0), 0)  # shape is 1, 2, N
    coord_uv = np.tile(coord_uv, [kp_uv.shape[0], 1, 1])  # shape is K, 2, N

    coord_uv -= np.expand_dims(kp_uv, -1)
    # dist = np.sqrt(np.sum(np.square(coord_uv), 1))  # shape is K, N: On second tought the sqrt is wrong ...
    dist = np.sum(np.square(coord_uv), 1)  # shape is K, N: On second tought the sqrt is wrong ...

    scoremap = np.exp(-dist / (sigma ** 2)) * vis
    scoremap = np.reshape(scoremap, (kp_uv.shape[0], img_shape_hw[0], img_shape_hw[1]))
    return np.transpose(scoremap, [1, 2, 0])


def create_scoremaps_2d_tf_batch(coords_uv, sigma, img_shape_hw, valid_vec=None):
    result = list()
    for bid in range(coords_uv.get_shape().as_list()[0]):
        result.append(create_scoremaps_2d_tf(coords_uv[bid],
                                             sigma,
                                             img_shape_hw,
                                             None if valid_vec is None else valid_vec[bid]))
    return tf.stack(result, 0)


def create_scoremaps_2d_tf(coords_uv, sigma, img_shape_hw, valid_vec=None):
    """ Creates a map of size (output_shape[0], output_shape[1]) at (center[0], center[1])
        with variance sigma for multiple coordinates."""
    with tf.name_scope('create_multiple_gaussian_map'):
        sigma = tf.cast(sigma, tf.float32)
        assert len(img_shape_hw) == 2
        s = coords_uv.get_shape().as_list()
        coords_uv = tf.cast(coords_uv, tf.int32)
        if valid_vec is not None:
            valid_vec = tf.cast(valid_vec, tf.float32)
            valid_vec = tf.squeeze(valid_vec)
            cond_val = tf.greater(valid_vec, 0.5)
        else:
            cond_val = tf.ones_like(coords_uv[:, 0], dtype=tf.float32)
            cond_val = tf.greater(cond_val, 0.5)

        cond_1_in = tf.logical_and(tf.less(coords_uv[:, 0], img_shape_hw[1] - 1), tf.greater(coords_uv[:, 0], 0))
        cond_2_in = tf.logical_and(tf.less(coords_uv[:, 1], img_shape_hw[0] - 1), tf.greater(coords_uv[:, 1], 0))
        cond_in = tf.logical_and(cond_1_in, cond_2_in)
        cond = tf.logical_and(cond_val, cond_in)

        coords_uv = tf.cast(coords_uv, tf.float32)

        # create meshgrid
        x_range = tf.expand_dims(tf.range(img_shape_hw[1]), 0)  # this is u
        y_range = tf.expand_dims(tf.range(img_shape_hw[0]), 1) # this is v

        X = tf.cast(tf.tile(x_range, [img_shape_hw[0], 1]), tf.float32)
        Y = tf.cast(tf.tile(y_range, [1, img_shape_hw[1]]), tf.float32)

        X.set_shape((img_shape_hw[0], img_shape_hw[1]))
        Y.set_shape((img_shape_hw[0], img_shape_hw[1]))

        X = tf.expand_dims(X, -1)
        Y = tf.expand_dims(Y, -1)

        X_b = tf.tile(X, [1, 1, s[0]])
        Y_b = tf.tile(Y, [1, 1, s[0]])

        X_b -= coords_uv[:, 0]
        Y_b -= coords_uv[:, 1]

        dist = tf.square(X_b) + tf.square(Y_b)

        scoremap = tf.exp(-dist / tf.square(sigma)) * tf.cast(cond, tf.float32)

        return scoremap


def create_scorevol_3d_tf_batch(coords_xyz, sigma, output_size, valid_vec=None):
    result = list()
    for bid in range(coords_xyz.get_shape().as_list()[0]):
        result.append(create_scorevol_3d_tf(coords_xyz[bid],
                                            sigma,
                                            output_size,
                                            None if valid_vec is None else valid_vec[bid]))
    return tf.stack(result, 0)


def create_scorevol_3d_tf(coords_xyz, sigma, output_size, valid_vec=None):
    """ Creates a map of size (output_shape[0], output_shape[1]) at (center[0], center[1])
        with variance sigma for multiple coordinates."""
    with tf.name_scope('create_multiple_gaussian_map'):
        assert len(output_size) == 3
        s = coords_xyz.get_shape().as_list()

        if valid_vec is not None:
            valid_vec = tf.cast(valid_vec, tf.float32)
            valid_vec = tf.squeeze(valid_vec)
            cond_val = tf.greater(valid_vec, 0.5)
        else:
            cond_val = tf.ones_like(coords_xyz[:, 0], dtype=tf.float32)
            cond_val = tf.greater(cond_val, 0.5)

        coords_xyz = tf.cast(coords_xyz, tf.float32)

        # create meshgrid
        x_range = tf.expand_dims(tf.expand_dims(tf.range(output_size[0]), -1), -1)
        y_range = tf.expand_dims(tf.expand_dims(tf.range(output_size[1]), 0), -1)
        z_range = tf.expand_dims(tf.expand_dims(tf.range(output_size[2]), 0), 0)

        X = tf.cast(tf.tile(x_range, [1, output_size[1], output_size[2]]), tf.float32)
        Y = tf.cast(tf.tile(y_range, [output_size[0], 1, output_size[2]]), tf.float32)
        Z = tf.cast(tf.tile(z_range, [output_size[0], output_size[1], 1]), tf.float32)

        X.set_shape((output_size[0], output_size[1], output_size[2]))
        Y.set_shape((output_size[0], output_size[1], output_size[2]))
        Z.set_shape((output_size[0], output_size[1], output_size[2]))

        X = tf.expand_dims(X, -1)
        Y = tf.expand_dims(Y, -1)
        Z = tf.expand_dims(Z, -1)

        X_b = tf.tile(X, [1, 1, 1, s[0]])
        Y_b = tf.tile(Y, [1, 1, 1, s[0]])
        Z_b = tf.tile(Z, [1, 1, 1, s[0]])

        X_b -= coords_xyz[:, 0]
        Y_b -= coords_xyz[:, 1]
        Z_b -= coords_xyz[:, 2]

        dist = tf.square(X_b) + tf.square(Y_b) + tf.square(Z_b)

        scoremap = tf.exp(-dist / (sigma ** 3)) * tf.cast(cond_val, tf.float32)
        return scoremap


def rank(tensor):
    # return the rank of a Tensor
    return len(tensor.get_shape())


def argmax_2d(tensor):
    with tf.name_scope('argmax_2d'):
        # input format: BxHxWxD
        assert rank(tensor) == 4

        # flatten the Tensor along the height and width axes
        flat_tensor = tf.reshape(tensor, (tf.shape(tensor)[0], -1, tf.shape(tensor)[3]))

        # argmax of the flat tensor
        argmax = tf.cast(tf.argmax(flat_tensor, axis=1), tf.int32)

        # convert indexes into 2D coordinates
        argmax_x = argmax // tf.shape(tensor)[2]
        argmax_y = argmax % tf.shape(tensor)[2]

        # stack and return 2D coordinates
        return tf.cast(tf.stack((argmax_x, argmax_y), axis=-1), tf.float32)


def argmax_3d(volume):
    with tf.name_scope('argmax_3d'):
        assert rank(volume) == 5
        s = volume.get_shape().as_list()
        volume = tf.reshape(volume, [s[0], -1, s[-1]])
        i = tf.argmax(volume, axis=1)

        x = i // (s[1]*s[2])
        y = (i - x*s[1]*s[2]) // s[1]
        z = (i - x*s[1]*s[2]) % s[1]
        return tf.cast(tf.stack([x, y, z], axis=-1), tf.float32)


if __name__ == '__main__':
    # volume = np.zeros((10, 10, 10))
    # # volume[7,3,1] = 1.0
    # volume[4,8,9] = 1.0
    # print(np.where(volume == volume.max()))
    #
    # volume_tf = tf.expand_dims(tf.expand_dims(tf.constant(volume), 0), -1)
    # ind, xyz = argmax_3d(volume_tf)
    # sess = tf.Session()
    # ind_v, xyz_v = sess.run([ind, xyz])
    # print('ind', ind_v.shape, ind_v)
    # print(xyz_v.shape, xyz_v)
    # exit()

    kp_uv = np.array([[140, 140.0],
                      [200, 200],
                      [50, 5]])

    # vis = np.array([1.0, 1.0, 1.0])

    scoremap = create_scoremaps_2d(kp_uv, sigma=10.0, img_shape_hw=(224, 224))

    # kp_uv = np.array([[14, 14.0],
    #                   [20, 20],
    #                   [5, 5]])
    #
    # # vis = np.array([1.0, 1.0, 1.0])
    #
    # scoremap = create_scoremaps_2d(kp_uv, sigma=2.0, img_shape_hw=(28, 28))

    from utils.mpl_setup import plt_figure
    plt, fig, axes = plt_figure(3)
    axes[0].imshow(scoremap[:, :, 0])
    axes[1].imshow(scoremap[:, :, 1])
    axes[2].imshow(scoremap[:, :, 2])
    plt.show()

    # kp_uv = np.array([[50, 50.0],
    #                   [190, 140],
    #                   [80, 20]])
    #
    # # vis = np.array([1.0, 1.0, 1.0])
    #
    # scoremap = create_scoremaps_2d(kp_uv, sigma=10.0, img_shape_hw=(150, 200))
    #
    # # TF version
    # kp_uv_tf = tf.constant(kp_uv)
    # scoremap_tf = create_scoremaps_2d_tf(kp_uv_tf, sigma=10.0, img_shape_hw=(150, 200))
    # sess = tf.Session()
    # scoremap_tf_v = sess.run(scoremap_tf)
    #
    # print('value at one sigma distance ', scoremap_tf_v[60, 50], )
    # print('value at two sigma distance', scoremap_tf_v[70, 50])
    # print('value at three sigma distance', scoremap_tf_v[80, 50])
    #
    # diff = np.abs(scoremap - scoremap_tf_v)
    # print('scoremap np', scoremap.min(), scoremap.max(), scoremap.shape)
    # print('scoremap tf', scoremap_tf_v.min(), scoremap_tf_v.max(), scoremap_tf_v.shape)
    # print('diff', diff.min(), diff.max())
    #
    # from utils.mpl_setup import plt_figure
    # plt, fig, axes = plt_figure(3)
    # axes[0].imshow(diff[:, :, 0])
    # axes[1].imshow(diff[:, :, 1])
    # axes[2].imshow(diff[:, :, 2])
    # plt.show(block=False)
    #
    # from utils.mpl_setup import plt_figure
    # plt, fig, axes = plt_figure(3)
    # axes[0].imshow(scoremap_tf_v[:, :, 0])
    # axes[1].imshow(scoremap_tf_v[:, :, 1])
    # axes[2].imshow(scoremap_tf_v[:, :, 2])
    # plt.show()
    #
    # coords_xyz = np.array([[3.0, 3.0, 3.0],
    #                        [12.0, 2.0, 3.0],
    #                        [6.0, 1.0, 3.0]])
    #
    # coords_xyz_tf = tf.constant(coords_xyz)
    # scorevol_tf = create_scorevol_3d_tf(coords_xyz_tf, output_size=(64, 64, 64), sigma=2.0)
    # sess = tf.Session()
    # scorevol_tf_v = sess.run(scorevol_tf)
    #
    # from utils.mpl_setup import plt_figure
    # plt, fig, axes = plt_figure(3)
    # axes[0].imshow(scorevol_tf_v[:, :, 3, 0])
    # axes[1].imshow(scorevol_tf_v[:, :, 3, 1])
    # axes[2].imshow(scorevol_tf_v[:, :, 3, 2])
    # plt.show(block=False)
    #
    # plt, fig, axes = plt_figure(3)
    # axes[0].imshow(scorevol_tf_v[:, :, 10, 0])
    # axes[1].imshow(scorevol_tf_v[:, :, 10, 1])
    # axes[2].imshow(scorevol_tf_v[:, :, 10, 2])
    # plt.show()

