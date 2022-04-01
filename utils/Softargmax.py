import tensorflow as tf


def softargmax2D_with_depthExtraction(scoremap, depthmap, output_px_space=True, return_maps=False):
    """ Given a scoremap it performs an softargmax on it. """
    with tf.variable_scope('Softargmax2DWithDepthExtraction'):
        s = scoremap.get_shape().as_list()
        assert len(s) == 4, "This was designed for predicted image like tensors [B, H, W, C] predictions."

        # Get meshgrid
        if output_px_space:
            range_x = tf.range(0, s[1])
            range_y = tf.range(0, s[2])
        else:
            range_x = tf.linspace(0.0, 1.0, s[1])
            range_y = tf.linspace(0.0, 1.0, s[2])
        H, W = tf.meshgrid(range_x, range_y, indexing='ij')
        H, W = tf.reshape(H, [1, -1, 1]), tf.reshape(W, [1, -1, 1])
        H, W = tf.cast(H, tf.float32), tf.cast(W, tf.float32)

        # 1. Softmax
        pred_reshaped = tf.reshape(scoremap, [s[0], -1, s[3]])  # reshape into a vector
        pred_softmaxed = tf.nn.softmax(pred_reshaped, axis=1)  # softmax over spatial dimension

        # 2. Accumulate coordinate
        h_pred = tf.reduce_sum(tf.multiply(H, pred_softmaxed), 1)
        w_pred = tf.reduce_sum(tf.multiply(W, pred_softmaxed), 1)
        pred_uv = tf.stack([w_pred, h_pred], -1)

        # 3. Get depth value
        depthmap_reshaped = tf.reshape(depthmap, [s[0], -1, s[3]])
        depthmap = tf.multiply(pred_softmaxed, depthmap_reshaped)
        depth = tf.reduce_sum(depthmap, 1)

        outputs = [pred_uv, tf.expand_dims(depth, -1)]
        if return_maps:
            outputs.append(tf.reshape(pred_softmaxed, [s[0], s[1], s[2], s[3]]))
            outputs.append(tf.reshape(depthmap, [s[0], s[1], s[2], s[3]]))

        return outputs


def softargmax2D(scoremap, output_px_space=True):
    """ Given a scoremap it performs an softargmax on it. """
    with tf.variable_scope('Softargmax2D'):
        s = scoremap.get_shape().as_list()
        assert len(s) == 4, "This was designed for predicted image like tensors [B, H, W, C] predictions."

        # Get meshgrid
        if output_px_space:
            range_x = tf.range(0, s[1])
            range_y = tf.range(0, s[2])
        else:
            range_x = tf.linspace(0.0, 1.0, s[1])
            range_y = tf.linspace(0.0, 1.0, s[2])
        H, W = tf.meshgrid(range_x, range_y, indexing='ij')
        H, W = tf.reshape(H, [1, -1, 1]), tf.reshape(W, [1, -1, 1])
        H, W = tf.cast(H, tf.float32), tf.cast(W, tf.float32)

        # 1. Softmax
        pred_reshaped = tf.reshape(scoremap, [s[0], -1, s[3]])  # reshape into a vector
        pred_softmaxed = tf.nn.softmax(pred_reshaped, axis=1)  # softmax over spatial dimension

        # 2. Accumulate coordinate
        h_pred = tf.reduce_sum(tf.multiply(H, pred_softmaxed), 1)
        w_pred = tf.reduce_sum(tf.multiply(W, pred_softmaxed), 1)
        pred_uv = tf.stack([w_pred, h_pred], -1)

        return pred_uv


def softmax2D(scoremap):
    """ Given a scoremap it performs an softargmax on it. """
    with tf.variable_scope('Softargmax2D'):
        s = scoremap.get_shape().as_list()
        assert len(s) == 4, "This was designed for predicted volumetric tensors [B, H, W, C] predictions. (Scorevolumes and such)."

        pred_reshaped = tf.reshape(scoremap, [s[0], -1, s[3]])  # reshape into a vector
        pred_softmaxed = tf.nn.softmax(pred_reshaped, axis=1)  # softmax over spatial dimension
        return tf.reshape(pred_softmaxed, s)


def softargmax3D(scorevolume, output_vox_space=True):
    """ Given a scorevolume it performs an softargmax on it. """
    with tf.variable_scope('Softargmax3D'):
        s = scorevolume.get_shape().as_list()
        assert len(s) == 5, "This was designed for predicted volumetric tensors [B, X, Y, Z, C] predictions. (Scorevolumes and such)."

        # Get meshgrid
        if output_vox_space:
            range_x = tf.range(0, s[1])
            range_y = tf.range(0, s[2])
            range_z = tf.range(0, s[3])
        else:
            range_x = tf.linspace(0.0, 1.0, s[1])
            range_y = tf.linspace(0.0, 1.0, s[2])
            range_z = tf.linspace(0.0, 1.0, s[3])
        X, Y, Z = tf.meshgrid(range_x, range_y, range_z, indexing='ij')
        X, Y, Z = tf.reshape(X, [1, -1, 1]), tf.reshape(Y, [1, -1, 1]), tf.reshape(Z, [1, -1, 1])
        X, Y, Z = tf.cast(X, tf.float32), tf.cast(Y, tf.float32), tf.cast(Z, tf.float32)

        # 1. Softmax
        pred_reshaped = tf.reshape(scorevolume, [s[0], -1, s[4]])  # reshape X, Y, Z into a vector
        pred_softmaxed = tf.nn.softmax(pred_reshaped, axis=1)  # softmax over spatial dimension

        # 2. Accumulate coordinate
        x_pred = tf.reduce_sum(tf.multiply(X, pred_softmaxed), 1)
        y_pred = tf.reduce_sum(tf.multiply(Y, pred_softmaxed), 1)
        z_pred = tf.reduce_sum(tf.multiply(Z, pred_softmaxed), 1)
        xyz_pred = tf.stack([x_pred, y_pred, z_pred], -1)

        return xyz_pred


def softargmax3D_w_uncertainty(scorevolume, output_vox_space=True):
    """ Given a scorevolume it performs an softargmax on it. """
    with tf.variable_scope('Softargmax3D'):
        s = scorevolume.get_shape().as_list()
        assert len(s) == 5, "This was designed for predicted volumetric tensors [B, X, Y, Z, C] predictions. (Scorevolumes and such)."

        # Get meshgrid
        if output_vox_space:
            range_x = tf.range(0, s[1])
            range_y = tf.range(0, s[2])
            range_z = tf.range(0, s[3])
        else:
            range_x = tf.linspace(0.0, 1.0, s[1])
            range_y = tf.linspace(0.0, 1.0, s[2])
            range_z = tf.linspace(0.0, 1.0, s[3])
        X, Y, Z = tf.meshgrid(range_x, range_y, range_z, indexing='ij')
        X, Y, Z = tf.reshape(X, [1, -1, 1]), tf.reshape(Y, [1, -1, 1]), tf.reshape(Z, [1, -1, 1])
        X, Y, Z = tf.cast(X, tf.float32), tf.cast(Y, tf.float32), tf.cast(Z, tf.float32)

        # 0. Split into coordinate part and uncertainty part
        scorevolume_kp = scorevolume[:, :, :, :, :s[4]//2]
        scorevolume_var = scorevolume[:, :, :, :, s[4]//2:]

        # 1. Softmax
        pred_reshaped = tf.reshape(scorevolume_kp, [s[0], -1, s[4]//2])  # reshape X, Y, Z into a vector
        pred_softmaxed = tf.nn.softmax(pred_reshaped, axis=1)  # softmax over spatial dimension

        # 2. Accumulate coordinate
        x_pred = tf.reduce_sum(tf.multiply(X, pred_softmaxed), 1)
        y_pred = tf.reduce_sum(tf.multiply(Y, pred_softmaxed), 1)
        z_pred = tf.reduce_sum(tf.multiply(Z, pred_softmaxed), 1)
        xyz_pred = tf.stack([x_pred, y_pred, z_pred], -1)

        # 3. Extract uncertainty from the locations
        pred_var_reshaped = tf.reshape(scorevolume_var, [s[0], -1, s[4]//2])  # reshape X, Y, Z into a vector
        # the softmaxed scores kind of select how we calculate the weighted average from the uncertainty volume
        uncertainty = tf.reduce_mean(tf.multiply(pred_var_reshaped, pred_softmaxed))

        return xyz_pred, uncertainty


def softargmax3D_w_single_uncertainty(scorevolume, output_vox_space=True):
    """ Given a scorevolume it performs an softargmax on it. """
    with tf.variable_scope('Softargmax3D'):
        s = scorevolume.get_shape().as_list()
        assert len(s) == 5, "This was designed for predicted volumetric tensors [B, X, Y, Z, C] predictions. (Scorevolumes and such)."

        # Get meshgrid
        if output_vox_space:
            range_x = tf.range(0, s[1])
            range_y = tf.range(0, s[2])
            range_z = tf.range(0, s[3])
        else:
            range_x = tf.linspace(0.0, 1.0, s[1])
            range_y = tf.linspace(0.0, 1.0, s[2])
            range_z = tf.linspace(0.0, 1.0, s[3])
        X, Y, Z = tf.meshgrid(range_x, range_y, range_z, indexing='ij')
        X, Y, Z = tf.reshape(X, [1, -1, 1]), tf.reshape(Y, [1, -1, 1]), tf.reshape(Z, [1, -1, 1])
        X, Y, Z = tf.cast(X, tf.float32), tf.cast(Y, tf.float32), tf.cast(Z, tf.float32)

        # 0. Split into coordinate part and uncertainty part (we have numkp + 1 chans and last one is joint uncertainty for this location)
        scorevolume_kp = scorevolume[:, :, :, :, :s[4]-1]
        scorevolume_var = scorevolume[:, :, :, :, -1:]

        # 1. Softmax
        pred_reshaped = tf.reshape(scorevolume_kp, [s[0], -1, s[4]-1])  # reshape X, Y, Z into a vector
        pred_softmaxed = tf.nn.softmax(pred_reshaped, axis=1)  # softmax over spatial dimension

        # 2. Accumulate coordinate
        x_pred = tf.reduce_sum(tf.multiply(X, pred_softmaxed), 1)
        y_pred = tf.reduce_sum(tf.multiply(Y, pred_softmaxed), 1)
        z_pred = tf.reduce_sum(tf.multiply(Z, pred_softmaxed), 1)
        xyz_pred = tf.stack([x_pred, y_pred, z_pred], -1)

        # 3. Extract uncertainty from the locations
        pred_var_reshaped = tf.reshape(scorevolume_var, [s[0], -1, 1])  # reshape X, Y, Z into a vector
        # the softmaxed scores kind of select how we calculate the weighted average from the uncertainty volume
        uncertainty = tf.reduce_mean(tf.multiply(pred_var_reshaped, pred_softmaxed), 1)  # something like std.dev
        uncertainty = tf.multiply(uncertainty, uncertainty)  # something like var

        return xyz_pred, uncertainty


def softmax3D(scorevolume):
    """ Normalizes a given volume into a valid distribution. """
    with tf.variable_scope('Softmax3D'):
        s = scorevolume.get_shape().as_list()
        assert len(s) == 5, "This was designed for predicted volumetric tensors [B, X, Y, Z, C] predictions. (Scorevolumes and such)."

        pred_reshaped = tf.reshape(scorevolume, [s[0], -1, s[4]])  # flatten X, Y, Z into single dimension
        pred_softmaxed = tf.nn.softmax(pred_reshaped, axis=1)  # softmax over spatial dimension

        return tf.reshape(pred_softmaxed, s)


if __name__ == '__main__':
    # import numpy as np
    #
    # scale = 0.0 # this scales the score
    # prediction = np.ones((1, 10, 10, 10, 2)) * -scale
    # prediction[0, 1, 2, 3, 0] = 1.0 * scale
    # prediction[0, 5, 5, 5, 1] = 1.0 * scale
    # prediction = tf.constant(prediction, dtype=tf.float32)
    #
    # xyz_pred = softargmax3D(prediction)
    # sess = tf.Session()
    # xyz_pred_v = sess.run(xyz_pred)
    # print('xyz_pred_v', xyz_pred_v)

    # Test meshgrid
    s = [1, 3, 3, 2]
    range_x = tf.range(0, s[1])
    range_y = tf.range(0, s[2])
    range_z = tf.range(0, s[3])
    X, Y, Z = tf.meshgrid(range_x, range_y, range_z, indexing='ij')
    # X, Y, Z = tf.reshape(X, [1, -1, 1]), tf.reshape(Y, [1, -1, 1]), tf.reshape(Z, [1, -1, 1])
    # X, Y, Z = tf.cast(X, tf.float32), tf.cast(Y, tf.float32), tf.cast(Z, tf.float32)

    sess = tf.Session()
    X, Y, Z = sess.run([X, Y, Z])
    print('X', X)
    print('Y', Y)
    print('Z', Z)