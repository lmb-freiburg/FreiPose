from __future__ import unicode_literals, print_function
import cv2
import numpy as np


def __get(coords, ind):
    if type(ind) == list:
        return np.mean(coords[ind], 0)
    return coords[ind]


def plot_origin(axis, d=0.1):
    corners = np.array([
        [0, 0, 0],
        [d, 0, 0],
        [0, d, 0],
        [0, 0, d],
    ])
    lines = [
        [0, 1],
        [0, 2],
        [0, 3],
    ]
    colors = ['r', 'g', 'b']
    for c, (pid, cid) in zip(colors, lines):
        axis.plot(corners[[pid, cid], 0], corners[[pid, cid], 1], corners[[pid, cid], 2], color=c, linewidth=1)  # ground plane


def plot_setup(axis, height=0.3):
    corners = np.array([
        [0.17590515376527321, 0.07319615152415435, 0.40260411374556704],
        [-0.23504167255991196, 0.06849604567623505, 0.5250420747005821],
        [-0.13562486117461803, 0.06455011084567093, 0.8563788839687796],
        [0.2723060115241149, 0.06945544062086159, 0.7313978578215554]
    ])
    lines = [
        [0, 1],
        [1, 2],
        [2, 3],
        [3, 0],
    ]
    for pid, cid in lines:
        axis.plot(corners[[pid, cid], 0], corners[[pid, cid], 1], corners[[pid, cid], 2], color='g', linewidth=1)  # ground plane
    for c in corners:
        axis.plot([c[0], c[0]],
                  [c[1], c[1] - height],
                  [c[2], c[2]],
                  color='r', linewidth=1)  # up direction


def plot_skel_3d(axis, model, coords_xyz, vis=None, color_fixed=None, linewidth='1', draw_kp=True):
    """ Plots a hand stick figure into a matplotlib figure. """
    if vis is None:
        vis = np.ones_like(coords_xyz[:, 0]) == 1.0

    for (pid, cid), color in model.limbs:
        color = [c/255.0 for c in color]

        if __get(vis, pid) < 1.0 or __get(vis, cid) < 1.0:
            continue
        coord1 = __get(coords_xyz, pid)
        coord2 = __get(coords_xyz, cid)

        coords = np.stack([coord1, coord2])
        if color_fixed is None:
            axis.plot(coords[:, 0], coords[:, 1], coords[:, 2], color=color, linewidth=linewidth)
        else:
            axis.plot(coords[:, 0], coords[:, 1], coords[:, 2], color_fixed, linewidth=linewidth)

    if not draw_kp:
        return

    for i in range(12):
        if vis[i]:
            axis.scatter(coords_xyz[i, 0], coords_xyz[i, 1], coords_xyz[i, 2], 'o', color=[0, 0 ,0])


def draw_skel(image, model, coords_hw, vis=None, color_fixed=None, linewidth=2, order='hw', img_order='rgb',
             draw_kp=True, kp_style=None,
             draw_limbs=True):
    """ Inpaints a hand stick figure into a matplotlib figure. """
    if kp_style is None:
        kp_style = (4, 1)

    image = np.squeeze(image)
    if len(image.shape) == 2:
        image = np.expand_dims(image, 2)
    s = image.shape
    assert len(s) == 3, "This only works for single images."

    convert_to_uint8 = False
    if s[2] == 1:
        # grayscale case
        image = (image - np.min(image)) / (np.max(image) - np.min(image) + 1e-4)
        image = np.tile(image, [1, 1, 3])
        pass
    elif s[2] == 3:
        # RGB case
        if image.dtype == np.uint8:
            convert_to_uint8 = True
            image = image.astype('float32') / 255.0
        elif image.dtype == np.float32:
            # convert to gray image
            image = np.mean(image, axis=2)
            image = (image - np.min(image)) / (np.max(image) - np.min(image) + 1e-4)
            image = np.expand_dims(image, 2)
            image = np.tile(image, [1, 1, 3])
    else:
        assert 0, "Unknown image dimensions."

    if order == 'uv':
        coords_hw = coords_hw[:, ::-1]

    color_map = {'k': np.array([0.0, 0.0, 0.0]),
                 'w': np.array([1.0, 1.0, 1.0]),
                 'b': np.array([0.0, 0.0, 1.0]),
                 'g': np.array([0.0, 1.0, 0.0]),
                 'r': np.array([1.0, 0.0, 0.0]),
                 'm': np.array([1.0, 1.0, 0.0]),
                 'c': np.array([0.0, 1.0, 1.0])}

    if vis is None:
        vis = np.ones_like(coords_hw[:, 0]) == 1.0

    if draw_limbs:
        for (pid, cid), color in model.limbs:
            if img_order == 'bgr':
                color = [color[2], color[1], color[0]]

            if __get(vis, pid) < 1.0 or __get(vis, cid) < 1.0:
                continue

            coord1 = __get(coords_hw, pid).astype(np.uint32)
            coord2 = __get(coords_hw, cid).astype(np.uint32)

            if (coord1[0] < 1) or (coord1[0] >= s[0]) or (coord1[1] < 1) or (coord1[1] >= s[1]):
                continue
            if (coord2[0] < 1) or (coord2[0] >= s[0]) or (coord2[1] < 1) or (coord2[1] >= s[1]):
                continue

            if color_fixed is None:
                color = [c/255.0 for c in color]
                cv2.line(image, (coord1[1], coord1[0]), (coord2[1], coord2[0]), color, thickness=linewidth)
            else:
                c = color_map.get(color_fixed, np.array([1.0, 1.0, 1.0]))
                if img_order == 'bgr':
                    c = [c[2], c[1], c[0]]
                cv2.line(image, (coord1[1], coord1[0]), (coord2[1], coord2[0]), c, thickness=linewidth)

    if draw_kp:
        coords_hw = coords_hw.astype(np.int32)
        for i, (_, color) in enumerate(model.keypoints):
            if vis[i]:
                if color_fixed is None:
                    color = [c/255.0 for c in color]
                    image = cv2.circle(image, (coords_hw[i, 1], coords_hw[i, 0]),
                                       radius=kp_style[0], color=color, thickness=kp_style[1])
                else:
                    c = color_map.get(color_fixed, np.array([1.0, 1.0, 1.0]))
                    image = cv2.circle(image, (coords_hw[i, 1], coords_hw[i, 0]),
                                       radius=kp_style[0], color=c, thickness=kp_style[1])

    if convert_to_uint8:
        image = (image * 255).astype('uint8')

    return image


def draw_trafo(img, M_trafo_w2l, M_cam_w2l, K_cam, dist_cam, linewidth=2, l=0.025):
    """ Draws a little coordinate frame into an image. """
    import utils.CamLib as cl

    M_trafo_l2w = np.linalg.inv(M_trafo_w2l)

    # points in local space we'd like to draw
    points_local = np.array([
        [0.0, 0.0, 0.0],  # origin
        [l, 0.0, 0.0],  # end x
        [0.0, l, 0.0],  #end y
        [0.0, 0.0, l]  #end z
    ])
    # trafo them to world space
    points_world = cl.trafo_coords(points_local, M_trafo_l2w)

    # transform points into image space
    p_cam = cl.trafo_coords(points_world, M_cam_w2l)
    p_uv = cl.project(p_cam, K_cam, dist_cam)
    p_uv = np.round(p_uv).astype(np.int32)

    # draw stuff
    bones, colors = [[0, 1], [0, 2], [0, 3]], [(0, 0, 255), (0, 255, 0), (255, 0, 0)]
    for b, c in zip(bones, colors):
        pid, cid = b
        img = cv2.line(img,
                       (p_uv[pid, 0], p_uv[pid, 1]),
                       (p_uv[cid, 0], p_uv[cid, 1]),
                       c, thickness=linewidth)
    return img


def draw_bb(image, bb, color=None, linewidth=8, mode='ltrb'):
    """ Inpaints a bounding box. """
    image = image.copy()
    bb = np.array(bb).copy()
    bb = np.round(bb).astype(np.int32)

    if color is None:
        color = 'r'

    color_map = {'k': np.array([0.0, 0.0, 0.0]),
                 'w': np.array([1.0, 1.0, 1.0]),
                 'b': np.array([0.0, 0.0, 1.0]),
                 'g': np.array([0.0, 1.0, 0.0]),
                 'r': np.array([1.0, 0.0, 0.0]),
                 'm': np.array([1.0, 1.0, 0.0]),
                 'c': np.array([0.0, 1.0, 1.0])}

    image = np.squeeze(image)
    if len(image.shape) == 2:
        image = np.expand_dims(image, 2)
    s = image.shape
    assert len(s) == 3, "This only works for single images."

    convert_to_uint8 = False
    if s[2] == 1:
        # grayscale case
        image = (image - np.min(image)) / (np.max(image) - np.min(image) + 1e-4)
        image = np.tile(image, [1, 1, 3])
        pass
    elif s[2] == 3:
        # RGB case
        if image.dtype == np.uint8:
            convert_to_uint8 = True
            image = image.astype('float32') / 255.0
        elif image.dtype == np.float32:
            # convert to gray image
            image = np.mean(image, axis=2)
            image = (image - np.min(image)) / (np.max(image) - np.min(image) + 1e-4)
            image = np.expand_dims(image, 2)
            image = np.tile(image, [1, 1, 3])
    else:
        assert 0, "Unknown image dimensions."

    # mapping = {'l': 0, 't': 1, 'r': 3, 'b': 4}
    mapping = {k: i for i, k in enumerate(mode)}

    points = list()
    points.append([bb[mapping['l']], bb[mapping['t']]])
    points.append([bb[mapping['l']], bb[mapping['b']]])
    points.append([bb[mapping['r']], bb[mapping['b']]])
    points.append([bb[mapping['r']], bb[mapping['t']]])
    points.append([bb[mapping['l']], bb[mapping['t']]])
    points = np.array(points)
    for i in range(len(points)-1):
        c = color_map.get(color, np.array([1.0, 0.0, 0.0]))
        cv2.line(image,
                 tuple(points[i]),
                 tuple(points[i+1]),
                 c, thickness=linewidth)

    if convert_to_uint8:
        image = (image * 255).astype('uint8')
    return image


def draw_text(img, text):
    s = np.max(img.shape[:2])
    m = int(round(0.2 * s))
    if s < 100:
        fontScale = 0.5
        thick = 2
    elif s < 200:
        fontScale = 1.0
        thick = 2
    elif s < 400:
        fontScale = 2.0
        thick = 3
    elif s < 800:
        fontScale = 4.0
        thick = 3
    else:
        fontScale = 6.0
        thick = 4

    img = cv2.putText(img, text, org=(m, m),
                fontFace=3, fontScale=fontScale,
                color=(0, 0, 255), thickness=thick)
    return img
