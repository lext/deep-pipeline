from functools import partial
from torchvision import transforms as tvt

import solt.core as slc
import solt.data as sld
import solt.transforms as slt

from deeppipeline.kvs import GlobalKVS
from deeppipeline.common.transforms import numpy2tens, apply_by_index


def img_binary_mask2solt(imgmask):
    img, mask = imgmask
    if len(img.shape) == 2:
        img = img.reshape(img.shape[0], img.shape[1], 1)
    return sld.DataContainer((img, mask.squeeze()), 'IM')


def solt2img_binary_mask(dc: sld.DataContainer):
    if dc.data_format != 'IM':
        raise ValueError
    if not isinstance(dc, sld.DataContainer):
        raise TypeError

    return dc.data[0], dc.data[1]


def init_binary_segmentation_augs():
    kvs = GlobalKVS()
    ppl = tvt.Compose([
        img_binary_mask2solt,
        slc.Stream([
            slt.PadTransform(pad_to=(kvs['args'].pad_x, kvs['args'].pad_y)),
            slt.RandomFlip(axis=1, p=0.5),
            slt.CropTransform(crop_size=(kvs['args'].crop_x, kvs['args'].crop_y), crop_mode='r'),
            slt.ImageGammaCorrection(gamma_range=(kvs['args'].gamma_min, kvs['args'].gamma_max), p=0.5),
        ]),
        solt2img_binary_mask,
        partial(apply_by_index, transform=numpy2tens, idx=[0, 1]),
    ])

    kvs.update('train_trf', ppl)

    return ppl


