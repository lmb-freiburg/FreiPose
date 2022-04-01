from tqdm import tqdm
import argparse, os
import numpy as np
import cv2, time
from scipy import signal

from config.Model import Model
from utils.general_util import json_load, json_dump
from utils.bodyvar_util import trafo_to_coord_frame, calculate_angle
import utils.CamLib as cl


def _trafo2local(kp_xyz):
    """ Transforms global keypoints into a rat local coordinate frame.

        The rat local system is spanned by:
            - x: Animal right  (perpendicular to ground plane normal and body axis)
            - y: The body axis (defined by the vector from tail to a point between the ears)
            - z: Animal up (perpendicular to x and y)
        And located in the point midway between the two ear keypoints.
    """
    mid_pt = 0.5 * (kp_xyz[5] + kp_xyz[0])  # point between ears
    body_axis = mid_pt - kp_xyz[11]  # vector from tail to mid ears, 'animal forward'
    body_axis /= np.linalg.norm(body_axis, 2, -1, keepdims=True)

    ground_up = np.array([0.0, -1.0, 0.0])  # vector pointing up
    ground_up /= np.linalg.norm(ground_up, 2)

    animal_right = np.cross(body_axis, ground_up)  # pointing into the animals' right direction
    animal_up = np.cross(animal_right, body_axis)

    R = np.stack([animal_right, body_axis, animal_up], 0)  # rotation matrix
    M = np.eye(4)
    M[:3, :3] = R
    M[:3, -1:] = -np.matmul(R, np.reshape(mid_pt, [3, 1]))  # trans
    kp_xyz_local = cl.trafo_coords(kp_xyz, M)
    return kp_xyz_local


def _calculate_local_positions(coord_def, pose_pred):
    """ Transform poses into rat local coordinate frame. """
    local_coords = [trafo_to_coord_frame(coord_def, p) for p in pose_pred]
    return np.stack(local_coords)


def _calculate_velocity(pose_pred):
    """ Calculate velocity as differences of neighboring frames. """
    vel = pose_pred[1:] - pose_pred[:-1]
    vel = np.concatenate([np.zeros_like(vel[:1]), vel], 0)
    return vel


def _fit_plane(points):
    """
    p, n = fit_plane(points)

    Given an array, points, of shape (d,...)
    representing points in d-dimensional space,
    fit an d-dimensional plane to the points.
    Return a point, p, on the plane (the point-cloud centroid),
    and the normal, n.
    """
    points = points.T
    points = np.reshape(points, (np.shape(points)[0], -1))  # Collapse trialing dimensions
    assert points.shape[0] <= points.shape[1], "There are only {} points in {} dimensions.".format(points.shape[1],
                                                                                                   points.shape[0])
    ctr = points.mean(axis=1)
    x = points - ctr[:, np.newaxis]
    M = np.dot(x, x.T)  # Could also use np.cov(x) here.
    return ctr, np.linalg.svd(M)[0][:, -1]


def _estimate_plane(plane_file):
    """ Estimates a plane (normal and point) from a given set of annotated points, assuming all point form a single plane. """
    plane_data = json_load(plane_file)
    plane = None
    for k, v in plane_data.items():
        if 'xyz' in v.keys() and 'vis3d' in v.keys():
            xyz = np.array(v['xyz'])
            vis3d = np.array(v['vis3d'])
            if np.sum(vis3d) >= 3:
                points = xyz[vis3d > 0.5]
                p, n = _fit_plane(points)

                if np.dot(n, np.array([0.0, -1.0, 0.0])) < 0.0:
                    # normal should roughly point in -y direction
                    n *= -1.0
                plane = p, n

    return plane


def _dist_point_plane(plane, p):
    """ Calculate distance point to plane. """
    x, n = plane
    x = np.reshape(x, [1, 3])
    n = np.reshape(n, [1, 3])
    return np.dot(p - x, n.T).squeeze()


def _calculate_plane_dist(pose_pred, plane):
    plane_dist = list()
    for p in pose_pred:
        dist = _dist_point_plane(plane, p)

        plane_dist.append(dist)
    return np.array(plane_dist)


def _angle_vec_plane(plane, vec):
    """ Calculate distance point to plane. """
    x, n = plane
    n = np.reshape(n, [3])
    n /= np.linalg.norm(n, 2)
    vec = np.reshape(vec, [3])
    vec /= np.linalg.norm(vec, 2)
    return np.arccos(np.dot(vec, n)).squeeze()


def _get_coord(id_list, xyz):
    if type(id_list) != list:
        id_list = [id_list]
    id_list = np.array(id_list).reshape([-1, 1])
    return np.mean(xyz[id_list], 0).squeeze()


def _calculate_plane_angle(pose_pred, plane, axis):
    plane_angle = list()
    for p in pose_pred:
        vec = _get_coord(axis[1], p) - _get_coord(axis[0], p)
        angle = _angle_vec_plane(plane, vec)
        plane_angle.append(angle)

    return np.array(plane_angle)


def _show_pairwise_dist(pairwise_dist, kp_pair_list, total_num_kp=12):
    from utils.mpl_setup import plt_figure
    dist_all = list()
    for vid_name, vid_data in pairwise_dist.items():
        for p in vid_data:
            if p is not None:
                dist_all.append(p)

    dist_all = np.array(dist_all)

    # figure out pairs
    kp_pair_list = [tuple(x) for x in kp_pair_list]
    cnt = 0
    show_tasks = list()
    for i in range(total_num_kp):
        for j in range(i+1, total_num_kp):
            if (i, j) in kp_pair_list:
                show_tasks.append(
                    [cnt, i, j]
                )
            cnt += 1

    plt, fig, axes = plt_figure(len(kp_pair_list))
    colors = ['r', 'g', 'b', 'c', 'm', 'k']
    for f, (c, (k, i, j)) in enumerate(zip(colors, show_tasks)):
        hist, edges = np.histogram(dist_all[:, k])
        hist = hist/float(np.sum(hist))
        bin_centers = 0.5*(edges[1:] + edges[:-1])
        axes[f].stem(bin_centers, hist, c, label='%d-%d' % (i, j))
        axes[f].legend()
    plt.show()


def read_vid_frame(video_path, fid):
    """ Reads a single frame from a video.
    """
    cap = cv2.VideoCapture(video_path)
    vid_size = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    assert 0 <= fid < vid_size, 'Frame id is outside the video.'

    cap.set(cv2.CAP_PROP_POS_FRAMES, fid)
    for i in range(5):
        suc, img = cap.read()
        if not suc:
            print('Reading video frame was not successfull. Will try again in 2 sec.')
            time.sleep(2)
        else:
            break
    assert img is not None and suc, 'Reading not successful'
    cap.release()
    return img

last_i = -1
def _show_coord_frame(args, name, pose_glob, pose_local):
    from utils.mpl_setup import  plt_figure
    from utils.plot_util import plot_skel_3d, plot_origin, plot_setup
    from matplotlib.widgets import Slider

    # set up figure
    num_fig = 3
    if args.video_file_name is not None:
        num_fig = 4
    plt, fig, axes = plt_figure(num_fig, is_3d_axis=[0, 1, 2])
    ax_slider = fig.add_axes([0.2, 0.05, 0.65, 0.03])  # left, bottom, width, height in fractions of figure w/h
    slider_ind = Slider(ax_slider, 'Fid', 0, pose_glob.shape[0], valinit=0, valfmt='%d')

    # update callback
    global last_i
    last_i = -1
    def _update(_):
        global last_i
        # get index for this sample
        i = int(slider_ind.val)

        if last_i == i:
            return
        last_i = i

        # clear content
        for ax in axes:
            ax.clear()

        skel = pose_glob[i].copy()
        plot_skel_3d(axes[0], model, skel), axes[0].set_title('global')
        plot_setup(axes[0])
        plot_origin(axes[0])
        axes[0].view_init(elev=-60, azim=-90)

        skel_n = pose_glob[i].copy()
        skel_n -= skel_n.mean(0, keepdims=True)
        plot_skel_3d(axes[1], model, skel_n), axes[1].set_title('global (centered)')
        plot_origin(axes[1])
        axes[1].view_init(elev=-60, azim=-90)

        skel_loc = pose_local[i].copy()
        plot_skel_3d(axes[2], model, skel_loc), axes[2].set_title('local: %s' % name)
        plot_origin(axes[2])

        for ax in [axes[1], axes[2]]:
            ax.set_xlim([-0.15, 0.15]), ax.set_ylim([-0.15, 0.15]), ax.set_zlim([-0.15, 0.15])
            ax.set_xlabel('x'), ax.set_ylabel('y'), ax.set_zlabel('z')

        if args.video_file_name is not None:
            img = read_vid_frame(args.video_file_name, i)
            axes[3].imshow(img[:, :, ::-1])
            axes[3].xaxis.set_visible(False), axes[3].yaxis.set_visible(False)
        fig.canvas.draw_idle()

    slider_ind.on_changed(_update)
    plt.show()


def _show_plane_dist(args, distances, pose_glob, kp_id_list, frame_diff=25):
    from utils.mpl_setup import  plt_figure
    from utils.plot_util import plot_skel_3d, plot_origin
    from matplotlib.widgets import Slider

    # set up figure
    num_fig = 2
    if args.video_file_name is not None:
        num_fig = 3
    plt, fig, axes = plt_figure(num_fig, is_3d_axis=[0])
    ax_slider = fig.add_axes([0.2, 0.05, 0.65, 0.03])  # left, bottom, width, height in fractions of figure w/h
    slider_ind = Slider(ax_slider, 'Fid', 0, pose_glob.shape[0]-frame_diff, valinit=0, valfmt='%d')

    # update callback
    global last_i
    last_i = -1
    def _update(_):
        global last_i
        # get index for this sample
        i = int(slider_ind.val)

        if last_i == i:
            return
        last_i = i

        # clear content
        for ax in axes:
            ax.clear()

        skel_n = pose_glob[i].copy()
        mean = skel_n.mean(0, keepdims=True)
        skel_n -= mean
        skel_n_t = pose_glob[i+frame_diff].copy() - mean
        plot_skel_3d(axes[0], model, skel_n, color_fixed='r')
        plot_skel_3d(axes[0], model, skel_n_t, color_fixed='g')
        axes[0].set_title('global (centered)')
        plot_origin(axes[0])
        axes[0].view_init(elev=-60, azim=-90)

        for j in kp_id_list:
            t = distances[i:(i+frame_diff), j]
            axes[1].plot(t, label='%d' % j)
        axes[1].legend()

        if args.video_file_name is not None:
            img = read_vid_frame(args.video_file_name, i)
            axes[2].imshow(img[:, :, ::-1])
            axes[2].xaxis.set_visible(False), axes[2].yaxis.set_visible(False)
        fig.canvas.draw_idle()

    slider_ind.on_changed(_update)
    plt.show()


def _show_plane_angles(args, angle_names, angles, pose_glob, frame_diff=25):
    from utils.mpl_setup import  plt_figure
    from utils.plot_util import plot_skel_3d, plot_origin
    from matplotlib.widgets import Slider

    # set up figure
    num_fig = 2
    if args.video_file_name is not None:
        num_fig = 3
    plt, fig, axes = plt_figure(num_fig, is_3d_axis=[0])
    ax_slider = fig.add_axes([0.2, 0.05, 0.65, 0.03])  # left, bottom, width, height in fractions of figure w/h
    slider_ind = Slider(ax_slider, 'Fid', 0, pose_glob.shape[0]-frame_diff, valinit=0, valfmt='%d')

    # update callback
    global last_i
    last_i = -1
    def _update(_):
        global last_i
        # get index for this sample
        i = int(slider_ind.val)

        if last_i == i:
            return
        last_i = i

        # clear content
        for ax in axes:
            ax.clear()

        skel_n = pose_glob[i].copy()
        mean = skel_n.mean(0, keepdims=True)
        skel_n -= mean
        skel_n_t = pose_glob[i+frame_diff].copy() - mean
        plot_skel_3d(axes[0], model, skel_n, color_fixed='r')
        plot_skel_3d(axes[0], model, skel_n_t, color_fixed='g')
        axes[0].set_title('global (centered)')
        plot_origin(axes[0])
        axes[0].view_init(elev=-60, azim=-90)

        for ax in [axes[0]]:
            ax.set_xlim([-0.15, 0.15]), ax.set_ylim([-0.15, 0.15]), ax.set_zlim([-0.15, 0.15])
            ax.set_xlabel('x'), ax.set_ylabel('y'), ax.set_zlabel('z')

        for n, a in zip(angle_names, angles):
            a = a.copy() * 180.0/np.pi
            axes[1].plot(a[i:(i+frame_diff)], label=n)
        axes[1].legend()

        if args.video_file_name is not None:
            img = read_vid_frame(args.video_file_name, i)
            axes[2].imshow(img[:, :, ::-1])
            axes[2].xaxis.set_visible(False), axes[2].yaxis.set_visible(False)
        fig.canvas.draw_idle()

    slider_ind.on_changed(_update)
    plt.show()


def _calc_avg_stft(time_signal, window_length=64):
    """ Calculates an STFT over the time signal and then averages across time segments.
        Returns an normalized signal of energy = 1
    """
    f, t, Zxx = signal.stft(time_signal, noverlap=window_length-1, nperseg=window_length, return_onesided=True)
    # Zxx = np.mean(np.abs(Zxx), 1)  # sum over time segments
    Zxx = np.abs(Zxx[:, 1:])  # use only norm and discard zero freq
    Zxx /= (1e-8 + np.sum(Zxx, 0, keepdims=True))  # normalize each time step to unit energy
    return Zxx


def _show_stft(args, stft, pose_glob, frame_diff=25):
    from utils.mpl_setup import plt_figure
    from utils.plot_util import plot_skel_3d, plot_origin
    from matplotlib.widgets import Slider

    # set up figure
    num_fig = 2
    if args.video_file_name is not None:
        num_fig = 3
    plt, fig, axes = plt_figure(num_fig, is_3d_axis=[0])
    ax_slider = fig.add_axes([0.2, 0.05, 0.65, 0.03])  # left, bottom, width, height in fractions of figure w/h
    slider_ind = Slider(ax_slider, 'Fid', 0, pose_glob.shape[0], valinit=0, valfmt='%d')

    # update callback
    global last_i
    last_i = -1
    def _update(_):
        global last_i
        # get index for this sample
        i = int(slider_ind.val)

        if last_i == i:
            return
        last_i = i

        # clear content
        for ax in axes:
            ax.clear()

        skel_n = pose_glob[i].copy()
        mean = skel_n.mean(0, keepdims=True)
        skel_n -= mean
        plot_skel_3d(axes[0], model, skel_n)
        axes[0].set_title('global (centered)')
        plot_origin(axes[0])
        axes[0].view_init(elev=-60, azim=-90)

        for ax in [axes[0]]:
            ax.set_xlim([-0.15, 0.15]), ax.set_ylim([-0.15, 0.15]), ax.set_zlim([-0.15, 0.15])
            ax.set_xlabel('x'), ax.set_ylabel('y'), ax.set_zlabel('z')

        s, e = i-frame_diff, i+frame_diff
        s, e = max(0, s), min(stft.shape[1]-1, e)
        axes[1].imshow(np.log(1+stft[:, s:e]))
        axes[1].set_xlabel('time'), axes[1].set_ylabel('freq')

        if args.video_file_name is not None:
            img = read_vid_frame(args.video_file_name, i)
            axes[2].imshow(img[:, :, ::-1])
            axes[2].xaxis.set_visible(False), axes[2].yaxis.set_visible(False)
        fig.canvas.draw_idle()

    slider_ind.on_changed(_update)
    plt.show()


def analyse(args, model, pose_pred):
    """ Function doing all the analysis. """
    variables = dict()  # output

    # turn into pose only array
    pose_pred = np.stack([np.array(p['kp_xyz'][0]) for p in pose_pred])

    if args.local:
        print('Calculating rat coordinates in a local coordinate frame.')
        for name, coord_def in model.coord_frames.items():
            data = _calculate_local_positions(coord_def, pose_pred)
            for i in range(data.shape[1]):
                # add distance/length as another variable
                dist = np.linalg.norm(data[:, i], 2, -1)
                variables['%s_%s_%s' % (name, model.keypoints[i][0], 'dist')] = dist
                for j, dim in enumerate(['x', 'y', 'z']):
                    variables['%s_%s_%s' % (name, model.keypoints[i][0], dim)] = data[:, i, j]
            print('Done with: %s' % name)

            if args.show:
                _show_coord_frame(args, name, pose_pred, data)

    if args.vel:
        print('Calculating rat velocity in a global frame.')
        variables['vel'] = _calculate_velocity(pose_pred)

    if args.vel_local:
        print('Calculating rat velocity in a local frame.')
        for frame_name, coord_def in model.coord_frames.items():
            for i in range(len(model.keypoints)):
                for j, dim in enumerate(['x', 'y', 'z', 'dist']):
                    name = '%s_%s_%s' % (frame_name, model.keypoints[i][0], dim)
                    variables['vel_%s' % name] = _calculate_velocity(variables[name])

    if args.plane:
        print('Calculating distance to a given plane of all rat keypoints.')
        plane = _estimate_plane(args.plane_file)
        data = _calculate_plane_dist(pose_pred, plane)
        # variables['plane_dist'] = _calculate_plane_dist(pose_pred, plane)

        for i in range(data.shape[1]):
            variables['%s_%s' % ('plane_dist', model.keypoints[i][0])] = data[:, i]

        if args.show:
            _show_plane_dist(args, data, pose_pred, kp_id_list=[2, 7])

    if args.plane_vel:
        print('Calculating rat velocity relative to a plane.')
        for i in range(len(model.keypoints)):
            kp_name = model.keypoints[i][0]
            variables['%s_%s' % ('plane_dist_vel', kp_name)] = _calculate_velocity(variables['%s_%s' % ('plane_dist', kp_name)])

    if args.body_angles:
        print('Calculating body angles.')
        angle_names, angle_data = list(), list() # for vis
        for name, angle_def in model.body_angles.items():
            data = np.array([calculate_angle(angle_def, p) for p in pose_pred])
            variables['%s_%s' % ('plane_angle', name)] = data
            print('Done with: ', name)
            angle_names.append( name )
            angle_data.append( data )

        if args.show:
            _show_plane_angles(args, angle_names, angle_data, pose_pred)

    if args.body_angle_vel:
        assert args.body_angles, 'These are needed.'
        print('Calculating velocity of body angles.')
        for name, angle_def in model.body_angles.items():
            angle_name = '%s_%s' % ('plane_angle', name)
            variables['vel_%s' % angle_name] = _calculate_velocity(variables[angle_name])

    if args.stft:
        print('Calculating STFT for all reasonable quantities.')
        for i in range(len(model.keypoints)):
            kp_name = model.keypoints[i][0]

            if args.plane:
                v = variables['%s_%s' % ('plane_dist', kp_name)]
                stft = _calc_avg_stft(v)
                for j, x in enumerate(stft):
                    variables['%s_%s_%d' % ('stft_plane', kp_name, j)] = x

            if args.local:
                for name, coord_def in model.coord_frames.items():
                    # for j, dim in enumerate(['x', 'y', 'z', 'dist']):
                    for j, dim in enumerate(['dist']):
                        v = variables['%s_%s_%s' % (name, kp_name, dim)]
                        stft = _calc_avg_stft(v)
                        for j, x in enumerate(stft):
                            variables['%s_%s_%s_%s_%d' % ('stft', name, kp_name, dim, j)] = x

        if args.show:
            v = variables['%s_%s' % ('plane_dist', 'Paw Front Right')]
            _show_stft(args, _calc_avg_stft(v), pose_pred)

    print('Calculated a total of %d body variables' % len(variables))
    return variables


if __name__ == '__main__':
    """ Converts a prediction file into behavioral variables.
    
    Example call:
    PRED_FILE="/misc/lmbraid18/zimmermc/datasets/RatTrack_storage_rat_neural_exp_al3_vid1/pred_unlabeled_pose_002.json"
    PLANE_FILE="/misc/lmbraid18/zimmermc/datasets/RatTrack_Laser/Rat512_20191008/labeled_set0/ground_plane.json"
    python pose_to_vars.py $PRED_FILE --save --vel --local --vel_local  --plane_file ${PLANE_FILE} --plane_vel --plane --body_angles --body_angles_vel --stft
    
    Which will create a file called "pred_pose_vars.json" in the path of $PRED_FILE, that is a dictionary with the keys being
    the variables indicated by the flags, i.e. if you call the script with --local, then there will be a "local" key in 
    it. Under every key there will be a key with the video's name which values are a list of the respective entity.
    """
    parser = argparse.ArgumentParser(description='Train network.')
    parser.add_argument('model', type=str, help='Model definition file.')
    parser.add_argument('pose_pred_file', type=str, help='Path to the pose file we want to analyse.')
    parser.add_argument('--file_out_name', type=str, default='pred_pose_vars.json', help='Name of the output file.')

    parser.add_argument('--local', action='store_true', help='If set, pose is returned in a rat local system.')
    parser.add_argument('--vel', action='store_true', help='If set, velocity is returned in the global coordinate system.')
    parser.add_argument('--vel_local', action='store_true', help='If set, velocity is returned in the rat local coordinate system.')
    parser.add_argument('--stft', action='store_true', help='If set, Calculates STFT..')

    parser.add_argument('--body_angles', action='store_true', help='If set, calculates angles of predefined body axes wrt ground plane.')
    parser.add_argument('--body_angle_vel', action='store_true', help='If set, calculates velocity of body angles.')

    parser.add_argument('--plane', action='store_true', help='If set, distance is returned relative to the the plane file.')
    parser.add_argument('--plane_vel', action='store_true', help='If set, calculates velocity relative to the given plane.')
    parser.add_argument('--plane_file', type=str, help='Path to the ground plane annotation file for the given sequence.')

    parser.add_argument('--show', action='store_true', help='If set, visualizes data.')
    parser.add_argument('--video_file_name', type=str, help='Video used for visualization.')
    parser.add_argument('--save', action='store_true', help='If set, saves data.')

    args = parser.parse_args()

    # load model data
    model = Model(args.model)

    # sanity check input
    assert os.path.exists(args.pose_pred_file), 'Given pose prediction file was not found.'
    if args.plane:
        assert os.path.exists(args.plane_file), 'Given plane definition file was not found.'

    # output file to save results to
    output_file_name = os.path.join(
        os.path.dirname(args.pose_pred_file),
        args.file_out_name
    )
    print('Output file: %s' % output_file_name)

    # load pose data
    pose_pred = json_load(args.pose_pred_file)

    # run analyse
    variables = analyse(args, model, pose_pred)

    if args.save:
        print('Saving file...')
        # save calculated variables
        json_dump(output_file_name, variables, verbose=True)
