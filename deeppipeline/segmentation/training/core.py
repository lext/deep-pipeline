import torch
from tqdm import tqdm
import gc
import numpy as np
import os
from tensorboardX import SummaryWriter
from termcolor import colored

from deeppipeline.common.core import save_checkpoint, init_optimizer, init_session
from deeppipeline.common.dataset import init_folds
from deeppipeline.kvs import GlobalKVS

from deeppipeline.segmentation.models import init_model
from deeppipeline.segmentation.losses import init_binary_loss
from deeppipeline.segmentation.training.dataset import init_segmentation_loaders
from deeppipeline.segmentation.training.dataset import init_data_processing

from deeppipeline.segmentation.evaluation import metrics
from deeppipeline.segmentation.evaluation.metrics import calculate_dice, calculate_iou
from deeppipeline.transforms.segmentation import init_binary_segmentation_augs

def pass_epoch(net, loader, optimizer, criterion):
    kvs = GlobalKVS()
    net.train(optimizer is not None)

    fold_id = kvs['cur_fold']
    epoch = kvs['cur_epoch']
    max_ep = kvs['args'].n_epochs
    n_classes = kvs['args'].n_classes

    running_loss = 0.0
    n_batches = len(loader)
    confusion_matrix = np.zeros((n_classes, n_classes), dtype=np.uint64)
    device = next(net.parameters()).device
    pbar = tqdm(total=n_batches, ncols=200)
    with torch.set_grad_enabled(optimizer is not None):
        for i, entry in enumerate(loader):
            if optimizer is not None:
                optimizer.zero_grad()

            inputs = entry['img'].to(device)
            mask = entry['mask'].to(device)
            outputs = net(inputs)
            loss = criterion(outputs, mask)

            if optimizer is not None:
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
                pbar.set_description(f"Fold [{fold_id}] [{epoch} | {max_ep}] | "
                                     f"Running loss {running_loss / (i + 1):.5f} / {loss.item():.5f}")
            else:
                running_loss += loss.item()
                pbar.set_description(desc=f"Fold [{fold_id}] [{epoch} | {max_ep}] | Validation progress")

                if n_classes == 2:
                    preds = outputs.gt(kvs['args'].binary_threshold)
                elif n_classes > 2:
                    preds = outputs.argmax(axis=1)
                else:
                    raise ValueError

                preds = preds.float().to('cpu').numpy()
                mask = mask.float().to('cpu').numpy()
                confusion_matrix += metrics.calculate_confusion_matrix_from_arrays(preds, mask, n_classes)

            pbar.update()
            gc.collect()
        gc.collect()
        pbar.close()

    return running_loss / n_batches, confusion_matrix


def log_metrics(writer, train_loss, val_loss, conf_matrix):
    kvs = GlobalKVS()

    dices = {'dice_{}'.format(cls): dice for cls, dice in enumerate(calculate_dice(conf_matrix))}
    ious = {'iou_{}'.format(cls): iou for cls, iou in enumerate(calculate_iou(conf_matrix))}
    print(colored('==> ', 'green') + 'Metrics:')
    print(colored('====> ', 'green') + 'Train loss:', train_loss)
    print(colored('====> ', 'green') + 'Val loss:', val_loss)
    print(colored('====> ', 'green') + f'Val Dice: {dices}')
    print(colored('====> ', 'green') + f'Val IoU: {ious}')
    dices_tb = {}
    for cls in range(1, len(dices)):
        dices_tb[f"Dice [{cls}]"] = dices[f"dice_{cls}"]

    ious_tb = {}
    for cls in range(1, len(ious)):
        ious_tb[f"IoU [{cls}]"] = ious[f"iou_{cls}"]

    to_log = {'train_loss': train_loss, 'val_loss': val_loss}
    # Tensorboard logging
    writer.add_scalars(f"Losses_{kvs['args'].model}", to_log, kvs['cur_epoch'])
    writer.add_scalars('Metrics/Dice', dices_tb, kvs['cur_epoch'])
    writer.add_scalars('Metrics/IoU', ious_tb, kvs['cur_epoch'])
    # KVS logging
    to_log.update({'epoch': kvs['cur_epoch']})
    val_metrics = {'epoch': kvs['cur_epoch']}
    val_metrics.update(to_log)
    val_metrics.update(dices)
    val_metrics.update({'conf_matrix': conf_matrix})

    kvs.update(f'losses_fold_[{kvs["cur_fold"]}]', to_log)
    kvs.update(f'val_metrics_fold_[{kvs["cur_fold"]}]', val_metrics)


def train_fold(net, train_loader, optimizer, criterion, val_loader, scheduler):
    kvs = GlobalKVS()
    fold_id = kvs['cur_fold']
    writer = SummaryWriter(os.path.join(kvs['args'].workdir, 'snapshots', kvs['snapshot_name'],
                                        'logs', 'fold_{}'.format(fold_id), kvs['snapshot_name']))

    for epoch in range(kvs['args'].n_epochs):
        print(colored('==> ', 'green') + f'Training epoch [{epoch}] with LR {scheduler.get_lr()}')
        kvs.update('cur_epoch', epoch)
        train_loss, _ = pass_epoch(net, train_loader, optimizer, criterion)
        val_loss, conf_matrix = pass_epoch(net, val_loader, None, criterion)
        log_metrics(writer, train_loss, val_loss, conf_matrix)
        save_checkpoint(net, optimizer, 'val_loss', 'lt')
        scheduler.step()


def train_n_folds(init_args, init_metadata, init_scheduler, img_reader,
                  mask_reader, init_augs=None, img_group_id_colname=None, img_class_colname=None, img_id_colname=None):

    args = init_args()
    kvs = init_session(args)[-1]
    init_metadata()
    assert 'metadata' in kvs
    if init_augs is None:
        if kvs['args'].n_classes == 2:
            init_augs = init_binary_segmentation_augs()
        else:
            raise NotImplementedError('Augmentations are not defined')
    else:
        init_augs()
    assert 'train_trf' in kvs
    init_data_processing(img_reader=img_reader, mask_reader=mask_reader)
    init_folds(img_group_id_colname=img_group_id_colname, img_class_colname=img_class_colname)

    for fold_id, x_train, x_val in kvs['cv_split']:
        kvs.update('cur_fold', fold_id)
        kvs.update('prev_model', None)

        net = init_model()
        optimizer = init_optimizer(net)
        if kvs['args'].n_classes == 2:
            criterion = init_binary_loss()
        else:
            raise NotImplementedError('Loss is not defined')
        scheduler = init_scheduler(optimizer)
        train_loader, val_loader = init_segmentation_loaders(x_train=x_train, x_val=x_val,
                                                             img_reader=img_reader,
                                                             mask_reader=mask_reader,
                                                             img_id_colname=img_id_colname,
                                                             img_group_id_colname=img_group_id_colname)

        train_fold(net=net, train_loader=train_loader,
                   optimizer=optimizer, criterion=criterion,
                   val_loader=val_loader, scheduler=scheduler)

