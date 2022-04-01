from __future__ import print_function, unicode_literals
import argparse

from config.Model import Model
from config.Param import Param
from pose_network.core.Trainer import Trainer
from pose_network.core.Types import *
from utils.general_util import find_first_non_existant


def setup_params(config, model):
    # derive some parameters from the config files
    sizes = [len(d['cam_range']) for d in model.datasets]
    assert all([sizes[0] == s for s in sizes]), 'Datasets have different number of cameras. This should not be the case.'
    config['batch_size'] = sizes[0]
    config['flag_step'] = config['pose_train_steps'] // 2
    config['num_kp'] = len(model.keypoints)
    config['job_name'] = find_first_non_existant('./trainings/%s' % config['pose_job_name'])

    config['limbs'] = [x[0] for x in model.limbs]  # toss away the limb colors
    config['kinematic_dep'] = model.data['kinematic_dep']
    config['limb_limits'] = model.data['limb_limits']
    config['limb_angle_limits'] = model.data['limb_angle_limits']


    # Only supervised training
    config['dataflows'] = [
            ('train_df', dataflow_t.rat_pose_mv, {
                'model': model,
                'datasets_list': config['db_train'],
                'is_train': True, 'threaded': True,
                'shuffle': True, 'single_sample': False,
                'voxel_dim': config['voxel_dim'], 'voxel_resolution': config['voxel_resolution']
            }
             ),
            ('test_df', dataflow_t.rat_pose_mv, {
                'model': model,
                'datasets_list': config['db_test'],
                'is_train': False, 'threaded': True,
                'shuffle': True, 'single_sample': False,
                'voxel_dim': config['voxel_dim'], 'voxel_resolution': config['voxel_resolution']
            }
             )
        ]

    # keep model around for plotting
    config['model'] = model

    # turn dictionary with keys into a class with fields
    class classify(object):
        def __init__(self, cfg):
            for k, v in cfg.items():
                setattr(self, k, v)

    return classify(config)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Start training of the pose network.')
    parser.add_argument('model', type=str, help='Model definition file.')
    args = parser.parse_args()

    # init model and config
    model = Model(args.model)
    param = Param()

    # some configs we have to setup
    config = setup_params(param.pose, model)

    # run training of pose estimation network
    trainer = Trainer(config)
    trainer.run()
