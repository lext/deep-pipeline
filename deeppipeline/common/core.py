import torch
from typing import Callable
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


def git_info():
    """
    Gathers info about the version of the codebase from git.

    Returns
    -------
    out : tuple of str
        branch and commit id
    """
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
    """
    Basic function that initializes each training loop.
    Sets the seed based on the parsed args, creates the snapshots dir and initializes global KVS.

    Parameters
    ----------
    args : Namespace
        Arguments from argparse.

    Returns
    -------
    out : tuple
        Args, snapshot name and global KVS.
    """
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
    """
    Initializes the optimizer for a given model.
    Currently supported optimizers are Adam and SGD with default parameters.
    Learning rate (LR) and weight decay (WD) must be specified in the arguments as `lr` and `wd`, respectively.
    LR and WD are retrieved automatically from global KVS.
    Parameters
    ----------
    net : torch.Module

    Returns
    -------
    out : torch.optim.Optimizer
        Initialized optimizer.

    """
    kvs = GlobalKVS()
    if kvs['args'].optimizer == 'adam':
        return optim.Adam(net.parameters(), lr=kvs['args'].lr, weight_decay=kvs['args'].wd)
    elif kvs['args'].optimizer == 'sgd':
        return optim.SGD(net.parameters(), lr=kvs['args'].lr, weight_decay=kvs['args'].wd, momentum=0.9)
    else:
        raise NotImplementedError


def log_metrics(writer, train_loss, val_loss, val_results, val_results_callback=None):
    """
    Basic function to log the results from the validation stage.
    takes Tensorboard writer, train loss, validation loss, the artifacts produced during the validation phase,
    and also additional callback that can process these data, e.g. compute the metrics and
    visualize them in Tensorboard. By default, train and validation losses are visualized outside of the callback.
    If any metric is computed in the callback, it is useful to log it into a dictionary `to_log`.



    Parameters
    ----------
    writer : SummaryWriter
        Tensorboard summary writer
    train_loss : float
        Training loss
    val_loss : float
        Validation loss
    val_results : object
        Artifacts produced during teh validation
    val_results_callback : Callable or None
        A callback function that can process the artifacts and e.g. display those in Tensorboard.

    Returns
    -------
    out : None

    """
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
    """
    Initializes a simple multi-step learning rate scheduler.
    The scheduling is done according to the scheduling parameters specified in the arguments.
    The parameter responsible for this is `lr_drop`.

    Parameters
    ----------
    optimizer : torch.optim.Optimizer
        Optimizer for which the scheduler need to be created
    Returns
    -------
    out : lr_scheduler.Scheduler
        Created Scheduler

    """
    kvs = GlobalKVS()
    return lr_scheduler.MultiStepLR(optimizer, kvs['args'].lr_drop)


def save_checkpoint(net, optimizer, val_metric_name, comparator='lt'):
    """
    Flexible function that saves the model and the optimizer states using a metric and a comparator.

    Parameters
    ----------
    net : torch.nn.Module
        Model
    optimizer : torch.optim.Optimizer
        Optimizer
    val_metric_name : str
        Name of the metric that needs to be used for snapshot comparison.
        This name needs match the once that were created in the callback function passed to
        `log_metrics`.
    comparator : str
        How to compare the previous and the current metric values - `lt` is less than, and `gt` is greater than.

    Returns
    -------
    out : None

    """
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
    """
    A common implementation of training one fold of a neural network. Presumably, it should be called
    within cross-validation loop.

    Parameters
    ----------
    pass_epoch : Callable
        Function that trains or validates one epoch
    net : torch.nn.Module
        Model to train
    train_loader : torch.utils.data.DataLoader
        Training data loader
    optimizer : torch.optim.Optimizer
        Optimizer
    criterion : torch.nn.Module
        Loss function
    val_loader : torch.utils.data.DataLoader
        Validation data loader
    scheduler : lr_scheduler.Scheduler
        Learning rate scheduler
    log_metrics_cb : Callable or None
        Callback that processes the artifacts from validation stage.

    Returns
    -------

    """
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
                  pass_epoch, log_metrics_cb,
                  img_group_id_colname=None, img_class_colname=None):
    """
    Full implementation of the n-fold training loop. Trains each fold in cross-validation.

    Parameters
    ----------
    init_args : Callable
        Function that initializes the arguments. This should store the arguments
        in global KVS as `args`.
    init_metadata : Callable
        Function that initializes metadata as a pandas dataframe. The function must store this object
        in global KVS under `metadata` tag.
    init_augs : Callable
        Must initialize the training data augmentations. Should store the callable transforms
        as `train_trf` in global KVS. Train transforms should be returned **without** mean-std normaliztion.
    init_data_processing : Callable
        Initializes the data processing. Should compute the mean and std or load those from file.
        Typically, this leverages the existing function from deep-pipeline see `deeppipeline.common.core.normaliztion`.
        After the mean and std are initialized they can be appended to `train_trf` and `train_trf` must be updated in
        global kvs. Thi function should also initialize the validation transforms.
    init_folds : Callable
        Initializes the n-fold cross validation depending on whether class and group for each image are specified.
    init_loaders : Callable
        Initializes the data loaders given the train and validation data splits stored as pandas data frames.
        This function is executed for every fold.
    init_model : Callable
        Initializes the model for every fold.
    init_loss : Callable
        Initializes the loss
    init_scheduler : Callable
        Initializes the scheduler
    pass_epoch : Callable
        Trains / validaties one epoch.
    log_metrics_cb : Callable or None
        Callback to process the artifacts of the validation function.
    img_group_id_colname : str or None
        Group id for each image. If not None, the folds are generated so that the images
        with the same group id cannot be in both train and validation simultaneously.
    img_class_colname : str or None
        Class for each image (if any specified). Used to stratify the train and the validation splits
        for every cross-validation fold.

    Returns
    -------
    out : None

    """

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

        train_fold(pass_epoch=pass_epoch, net=net, train_loader=train_loader,
                   optimizer=optimizer, criterion=criterion,
                   val_loader=val_loader, scheduler=scheduler, log_metrics_cb=log_metrics_cb)
