from torch import nn
from deeppipeline.kvs import GlobalKVS
from ._unet import UNet


def init_model(ignore_data_parallel=False):
    kvs = GlobalKVS()
    if kvs['args'].model == 'unet':
        net = UNet(bw=kvs['args'].bw, depth=kvs['args'].depth,
                   center_depth=kvs['args'].cdepth,
                   n_inputs=kvs['args'].n_inputs,
                   n_classes=kvs['args'].n_classes - 1,
                   activation='relu')
    else:
        raise NotImplementedError

    if not ignore_data_parallel:
        if kvs['gpus'] > 1:
            net = nn.DataParallel(net).to('cuda')

    net = net.to('cuda')

    return net
