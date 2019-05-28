import torch
import numpy as np
import time
from termcolor import colored
import subprocess
import socket
from torch import optim
import os
import operator
from deeppipeline.kvs import GlobalKVS
from tensorboardX import SummaryWriter
from torch.optim import lr_scheduler


# Return the git revision as a string
def git_info():
    def _minimal_ext_cmd(cmd):
        # construct minimal environment
        env = {}
        for k in ['SYSTEMROOT', 'PATH']:
            v = os.environ.get(k)
            if v is not None:
                env[k] = v
        # LANGUAGE is used on win32
        env['LANGUAGE'] = 'C'
        env['LANG'] = 'C'
        env['LC_ALL'] = 'C'
        return subprocess.Popen(cmd, stdout=subprocess.PIPE, env=env).communicate()[0]

    try:
        out = _minimal_ext_cmd(['git', 'rev-parse', 'HEAD'])
        git_revision = out.strip().decode('ascii')

        out = _minimal_ext_cmd(['git', 'rev-parse', '--abbrev-ref', 'HEAD'])
        git_branch = out.strip().decode('ascii')
    except OSError:
        return None

    return git_branch, git_revision


def init_session(args):
    # Initializing the seeds
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)
    # Creating the snapshot
    snapshot_name = time.strftime(f'{socket.gethostname()}_%Y_%m_%d_%H_%M')
    os.makedirs(os.path.join(args.workdir, 'snapshots',  snapshot_name), exist_ok=True)

    kvs = GlobalKVS(os.path.join(args.workdir, 'snapshots', snapshot_name, 'session.pkl'))

    res = git_info()
    if res is not None:
        kvs.update('git branch name', res[0])
        kvs.update('git commit id', res[1])
    else:
        kvs.update('git branch name', None)
        kvs.update('git commit id', None)

    kvs.update('pytorch_version', torch.__version__)

    if torch.cuda.is_available():
        kvs.update('cuda', torch.version.cuda)
        kvs.update('gpus', torch.cuda.device_count())
    else:
        kvs.update('cuda', None)
        kvs.update('gpus', None)

    kvs.update('snapshot_name', snapshot_name)
    kvs.update('args', args)

    return args, snapshot_name, kvs


def init_optimizer(net):
    kvs = GlobalKVS()
    if kvs['args'].optimizer == 'adam':
        return optim.Adam(net.parameters(), lr=kvs['args'].lr, weight_decay=kvs['args'].wd)
    elif kvs['args'].optimizer == 'sgd':
        return optim.SGD(net.parameters(), lr=kvs['args'].lr, weight_decay=kvs['args'].wd, momentum=0.9)
    else:
        raise NotImplementedError


def log_metrics(writer, train_loss, val_loss, val_results, val_results_callback=None):
    kvs = GlobalKVS()

    print(colored('==> ', 'green') + 'Metrics:')
    print(colored('====> ', 'green') + 'Train loss:', train_loss)
    print(colored('====> ', 'green') + 'Val loss:', val_loss)

    to_log = {'train_loss': train_loss, 'val_loss': val_loss}
    val_metrics = {'epoch': kvs['cur_epoch']}
    val_metrics.update(to_log)
    writer.add_scalars(f"Losses_{kvs['args'].annotations}", to_log, kvs['cur_epoch'])
    if val_results_callback is not None:
        val_results_callback(writer, val_metrics, to_log, val_results)

    kvs.update(f'losses_fold_[{kvs["cur_fold"]}]', to_log)
    kvs.update(f'val_metrics_fold_[{kvs["cur_fold"]}]', val_metrics)


def init_ms_scheduler(optimizer):
    kvs = GlobalKVS()
    return lr_scheduler.MultiStepLR(optimizer, kvs['args'].lr_drop)


def save_checkpoint(net, optimizer, val_metric_name, comparator='lt'):
    if isinstance(net, torch.nn.DataParallel):
        net = net.module

    kvs = GlobalKVS()
    fold_id = kvs['cur_fold']
    epoch = kvs['cur_epoch']
    val_metric = kvs[f'val_metrics_fold_[{fold_id}]'][-1][0][val_metric_name]
    comparator = getattr(operator, comparator)
    cur_snapshot_name = os.path.join(os.path.join(kvs['args'].workdir, 'snapshots', kvs['snapshot_name'],
                                     f'fold_{fold_id}_epoch_{epoch}.pth'))

    if kvs['prev_model'] is None:
        print(colored('====> ', 'red') + 'Snapshot was saved to', cur_snapshot_name)
        torch.save({'model': net.state_dict(), 'optimizer': optimizer.state_dict()}, cur_snapshot_name)
        kvs.update('prev_model', cur_snapshot_name)
        kvs.update('best_val_metric', val_metric)

    else:
        if comparator(val_metric, kvs['best_val_metric']):
            print(colored('====> ', 'red') + 'Snapshot was saved to', cur_snapshot_name)
            os.remove(kvs['prev_model'])
            torch.save({'model': net.state_dict(), 'optimizer': optimizer.state_dict()}, cur_snapshot_name)
            kvs.update('prev_model', cur_snapshot_name)
            kvs.update('best_val_metric', val_metric)


def train_fold(pass_epoch, net, train_loader, optimizer, criterion, val_loader, scheduler, log_metrics_cb=None):
    kvs = GlobalKVS()
    fold_id = kvs['cur_fold']
    writer = SummaryWriter(os.path.join(kvs['args'].workdir, 'snapshots', kvs['snapshot_name'],
                                        'logs', 'fold_{}'.format(fold_id), kvs['snapshot_name']))

    for epoch in range(kvs['args'].n_epochs):
        print(colored('==> ', 'green') + f'Training epoch [{epoch}] with LR {scheduler.get_lr()}')
        kvs.update('cur_epoch', epoch)
        train_loss, _ = pass_epoch(net, train_loader, optimizer, criterion)
        val_loss, val_results = pass_epoch(net, val_loader, None, criterion)
        log_metrics(writer, train_loss, val_loss, val_results, log_metrics_cb)
        save_checkpoint(net, optimizer, 'val_loss', 'lt')
        scheduler.step()


def train_n_folds(init_args, init_metadata, init_augs,
                  init_data_processing,
                  init_folds,
                  init_loaders,
                  init_model, init_loss,
                  init_scheduler,
                  img_group_id_colname=None, img_class_colname=None):

    args = init_args()
    kvs = init_session(args)[-1]
    init_metadata()
    assert 'metadata' in kvs

    if init_augs is None:
        raise NotImplementedError('Train augmentations are not defined !!!')
    else:
        init_augs()
    assert 'train_trf' in kvs
    init_data_processing()
    init_folds(img_group_id_colname=img_group_id_colname, img_class_colname=img_class_colname)

    for fold_id, x_train, x_val in kvs['cv_split']:
        kvs.update('cur_fold', fold_id)
        kvs.update('prev_model', None)

        net = init_model()
        optimizer = init_optimizer(net)
        criterion = init_loss()
        scheduler = init_scheduler(optimizer)
        train_loader, val_loader = init_loaders(x_train, x_val)

        train_fold(net=net, train_loader=train_loader,
                   optimizer=optimizer, criterion=criterion,
                   val_loader=val_loader, scheduler=scheduler)
