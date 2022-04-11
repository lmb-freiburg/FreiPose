from pose_network.core.Types import *
from .RatNetMultiView import *


def create_networks(config):
    """ Creates networks. """
    arches = config.architectures
    if not type(arches) == list:
        arches = [arches]

    network_list = list()
    for arch in arches:
        if arch == network_t.RatNetMV:
            network_list.append(RatNetMultiView(config))

        else:
            print('architectures:', arch)
            raise NotImplementedError
    return network_list
