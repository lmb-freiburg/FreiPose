from __future__ import print_function, unicode_literals

from pose_network.core.Types import *


def create_dataflow(name, flow_type):
    if flow_type == dataflow_t.rat_pose_mv:
        from datareader.reader_labeled import build_dataflow, df2dict

    elif flow_type == dataflow_t.rat_pose_mv_semi:
        from datareader.reader_semilabeled import build_dataflow, df2dict

    else:
        print('name, flow_type', name, flow_type)
        raise NotImplementedError

    return df2dict, build_dataflow