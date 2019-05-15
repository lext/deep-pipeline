import torch
from torch.utils import data
import numpy as np

import os
from functools import partial
from torchvision import transforms as tvt

from deeppipeline.io import read_rgb_ocv, read_gs_binary_mask_ocv
from deeppipeline.kvs import GlobalKVS
from deeppipeline.common.normalization import init_mean_std, normalize_channel_wise
from deeppipeline.common.transforms import apply_by_index, numpy2tens


class SegmentationDataset(data.Dataset):
    def __init__(self, split, trf, read_img, read_mask, img_id_colname=None, img_group_id_colname=None):
        self.split = split
        self.transforms = trf
        self.read_img = read_img
        self.read_mask = read_mask
        self.img_id_colname = img_id_colname
        self.img_group_id_colname = img_group_id_colname

    def __getitem__(self, idx):
        entry = self.split.iloc[idx]
        img_fname = entry.img_fname
        mask_fname = entry.mask_fname

        img = self.read_img(img_fname)
        mask = self.read_mask(mask_fname)

        img, mask = self.transforms((img, mask))

        res = {'img': img, 'mask': mask, 'fname': img_fname.split('/')[-1]}
        if self.img_id_colname is not None:
            res['img_id'] = getattr(entry, self.img_id_colname)

        if self.img_group_id_colname is not None:
            res['group_id'] = getattr(entry, self.img_group_id_colname)

        return res

    def __len__(self):
        return self.split.shape[0]


def init_data_processing(img_reader=read_rgb_ocv, mask_reader=read_gs_binary_mask_ocv):
    kvs = GlobalKVS()

    dataset = SegmentationDataset(split=kvs['metadata'],
                                  trf=kvs['train_trf'],
                                  read_img=img_reader,
                                  read_mask=mask_reader)

    tmp = init_mean_std(snapshots_dir=os.path.join(kvs['args'].workdir, 'snapshots'),
                        dataset=dataset,
                        batch_size=kvs['args'].bs,
                        n_threads=kvs['args'].n_threads,
                        n_classes=kvs['args'].n_classes)

    if len(tmp) == 3:
        mean_vector, std_vector, class_weights = tmp
    elif len(tmp) == 2:
        mean_vector, std_vector = tmp
    else:

        raise ValueError('Incorrect format of mean/std/class-weights')

    norm_trf = partial(normalize_channel_wise, mean=mean_vector, std=std_vector)

    train_trf = tvt.Compose([
        kvs['train_trf'],
        partial(apply_by_index, transform=norm_trf, idx=0)
    ])

    val_trf = tvt.Compose([
        partial(apply_by_index, transform=numpy2tens, idx=[0, 1]),
        partial(apply_by_index, transform=norm_trf, idx=0)
    ])
    kvs.update('class_weights', class_weights)
    kvs.update('train_trf', train_trf)
    kvs.update('val_trf', val_trf)


def init_segmentation_loaders(x_train, x_val, img_reader=read_rgb_ocv, mask_reader=read_gs_binary_mask_ocv,
                              img_id_colname=None, img_group_id_colname=None):
    kvs = GlobalKVS()

    train_dataset = SegmentationDataset(split=x_train,
                                        trf=kvs['train_trf'],
                                        read_img=img_reader,
                                        read_mask=mask_reader,
                                        img_id_colname=img_id_colname,
                                        img_group_id_colname=img_group_id_colname)

    val_dataset = SegmentationDataset(split=x_val,
                                      trf=kvs['val_trf'],
                                      read_img=img_reader,
                                      read_mask=mask_reader,
                                      img_id_colname=img_id_colname,
                                      img_group_id_colname=img_group_id_colname)

    train_loader = data.DataLoader(train_dataset, batch_size=kvs['args'].bs,
                                   num_workers=kvs['args'].n_threads, shuffle=True,
                                   drop_last=True,
                                   worker_init_fn=lambda wid: np.random.seed(np.uint32(torch.initial_seed() + wid)))

    val_loader = data.DataLoader(val_dataset, batch_size=kvs['args'].val_bs,
                                 num_workers=kvs['args'].n_threads)

    return train_loader, val_loader

