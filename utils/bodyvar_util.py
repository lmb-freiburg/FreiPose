import numpy as np
import utils.CamLib as cl


def _get_coord(id_list, xyz):
    id_list = np.array(id_list).reshape([-1, 1])
    return np.mean(xyz[id_list], 0).squeeze()


def _get_origin(origin_def, xyz):
    """ Get location from an definition. """
    mode, definition = origin_def
    if mode == 'l':
        return _get_coord(definition, xyz)

    elif mode == 'g':
        return np.array(definition)


def _get_vec(axes_def, xyz):
    """ Get axis from an definition. """
    mode, definition = axes_def
    if mode == 'l':
        p = _get_coord(definition[0], xyz)
        c = _get_coord(definition[1], xyz)
        vec = c - p

    elif mode == 'g':
        vec = np.array(definition)

    else:
        raise NotImplementedError('Invalid mode.')

    return np.reshape(vec, [3])


def _norm(vec):
    """ L2 norm of a vector. """
    n = np.linalg.norm(vec, 2)
    if n < 1e-6:
        raise Exception('Small vector used for axis definition.')
    return vec/n


def _to_mat(x_axis, y_axis, z_axis, origin):
    R = np.stack([x_axis, y_axis, z_axis], 0)  # rotation matrix
    M = np.eye(4)
    M[:3, :3] = R
    M[:3, -1:] = -np.matmul(R, np.reshape(origin, [3, 1]))  # trans
    return M


def _my_cross(x=None, y=None, z=None):
    """ Calculate cross product without having to worry about getting the order and signs right. """
    if x is not None and y is not None:
        return np.cross(x, y)
    elif x is not None and z is not None:
        return np.cross(x, -z)
    elif y is not None and z is not None:
        return np.cross(y, z)
    else:
        raise NotImplementedError('This should never happen.')


def trafo_to_coord_frame(coord_def, xyz):
    """ Calculate the transformation matrix. """
    ori = coord_def['orientation']
    has_x = 'x' in ori.keys()
    has_y = 'y' in ori.keys()
    has_z = 'z' in ori.keys()
    assert sum([has_x, has_y, has_z]) == 2, 'You get to chose exactly two axes, no more, no less.'

    if has_x and has_y:
        x_vec = _norm(_get_vec(ori['x'], xyz))
        y_vec = _norm(_get_vec(ori['y'], xyz))
        z_vec = _my_cross(x=x_vec, y=y_vec)

        if ori['prio'] == 'x':
            y_vec = _my_cross(x=x_vec, z=z_vec)
        elif ori['prio'] == 'y':
            x_vec = _my_cross(y=y_vec, z=z_vec)
        else:
            raise NotImplementedError('This should never happen.')

    elif has_x and has_z:
        x_vec = _norm(_get_vec(ori['x'], xyz))
        z_vec = _norm(_get_vec(ori['z'], xyz))
        y_vec = _my_cross(x=x_vec, z=z_vec)

        if ori['prio'] == 'x':
            z_vec = _my_cross(x=x_vec, y=y_vec)
        elif ori['prio'] == 'z':
            x_vec = _my_cross(y=y_vec, z=z_vec)
        else:
            raise NotImplementedError('This should never happen.')

    elif has_y and has_z:
        y_vec = _norm(_get_vec(ori['y'], xyz))
        z_vec = _norm(_get_vec(ori['z'], xyz))
        x_vec = _my_cross(y=y_vec, z=z_vec)

        if ori['prio'] == 'y':
            z_vec = _my_cross(x=x_vec, y=y_vec)
        elif ori['prio'] == 'z':
            y_vec = _my_cross(x=x_vec, z=z_vec)
        else:
            raise NotImplementedError('This should never happen.')

    else:
        raise NotImplementedError('This should never happen.')

    origin = _get_origin(coord_def['origin'], xyz)
    M = _to_mat(x_vec, y_vec, z_vec, origin)
    xyz_local = cl.trafo_coords(xyz, M)
    return xyz_local


def calculate_angle(angle_def, xyz):
    """ Calculate angle in rad between two vectors. """
    vec1 = _norm(_get_vec(angle_def[0], xyz))
    vec2 = _norm(_get_vec(angle_def[1], xyz))
    return np.arccos(np.dot(vec1, vec2))
