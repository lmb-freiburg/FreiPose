import argparse, os, shutil
from config.Model import Model

from utils.general_util import my_mkdir, sort_nicely


def _is_image_file(f):
    good_ext = ['png', 'jpg', 'bmp']
    ext = os.path.splitext(f)[1].lower()

    if ext[1:] in good_ext:
        return True
    return False


def export(output_path, db_id, db):
    def _join_check(*args):
        p = os.path.join(*args)
        assert os.path.exists(p), 'File should exist.'
        return p

    output_path_this = os.path.join(output_path, 'run%03d' % db_id)
    calib_file = _join_check(db['path'], db['calib'])
    anno_file = _join_check(db['path'], db['frame_dir'], db['anno'])
    frame_dir = _join_check(db['path'], db['frame_dir'])

    # copy frames
    print('Dealing with', frame_dir, ' saved to:', output_path_this)
    for cid in db['cam_range']:
        print('cam %d/%d' % (cid, len(db['cam_range'])), end='\r')

        output_path_this_cam = os.path.join(output_path_this, 'cam%d' % cid)
        frame_dir_this_cam = os.path.join(frame_dir, 'cam%d' % cid)
        frames = os.listdir(frame_dir_this_cam)
        frames = [os.path.join(frame_dir, 'cam%d' % cid, f) for f in frames]
        frames = [f for f in frames if _is_image_file(f)]
        sort_nicely(frames)
        assert len(frames) > 0, 'There should be frames.'

        my_mkdir(output_path_this_cam, is_file=False)

        for i, f in enumerate(frames):
            shutil.copy2(f,
                         os.path.join(output_path_this_cam))

    my_mkdir(output_path_this, is_file=False)
    shutil.copy2(calib_file,
                 os.path.join(output_path_this, 'M.json'))
    shutil.copy2(anno_file,
                 os.path.join(output_path_this, 'anno.json'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Accumulate all labeled data into a single dataset.')
    parser.add_argument('model', type=str, help='Model definition file.')
    parser.add_argument('output_path', type=str, help='Path to where to save the data.')
    parser.add_argument('--set_name', type=str, default='train', help='Set name to export.')
    parser.add_argument('--force', action='store_true', help='Force outputting even when the path already exists.')
    args = parser.parse_args()

    # create output path
    if not args.force:
        assert not os.path.exists(args.output_path), 'Path should not exist yet.'
    my_mkdir(args.output_path, is_file=False)

    m = Model(args.model)

    # output
    db_id = 0
    for db in m.datasets:
        if db['db_set'] != args.set_name:
            continue

        # export
        export(args.output_path, db_id, db)
        db_id += 1