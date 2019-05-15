import torch
from tqdm import tqdm
import gc
import numpy as np

from termcolor import colored

from deeppipeline.kvs import GlobalKVS
from deeppipeline.segmentation.evaluation import metrics
from deeppipeline.segmentation.evaluation.metrics import calculate_dice, calculate_iou


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
                mask = mask.to('cpu').numpy()
                if n_classes == 2:
                    preds = (outputs.to('cpu').numpy() > kvs['args'].binary_threshold).astype(float)
                elif n_classes > 2:
                    preds = outputs.to('cpu').numpy().argmax(axis=1)
                else:
                    raise ValueError
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



