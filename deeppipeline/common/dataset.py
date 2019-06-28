import os

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, GroupKFold, StratifiedKFold

from deeppipeline.kvs import GlobalKVS


def init_folds(img_group_id_colname=None, img_class_colname=None):
    """
    Initialzies the cross-validation splits.

    Parameters
    ----------
    img_group_id_colname : str or None
        Column in `metadata` that is used to create cross-validation splits.
        If not None, then images that have the same group_id are never in train and validation.
    img_class_colname : str or None
        Column in `metadata` that is used to create cross-validation splits. If not none,
        splits are stratifed to ensure the same distribution of `img_class_colname` in train and validation.

    Returns
    -------

    """
    kvs = GlobalKVS()

    if img_group_id_colname is not None:
        gkf = GroupKFold(kvs['args'].n_folds)
        if img_class_colname is not None:
            class_col_name = getattr(kvs['metadata'], img_class_colname, None)
        else:
            class_col_name = None
        splitter = gkf.split(X=kvs['metadata'],
                             y=class_col_name,
                             groups=getattr(kvs['metadata'], img_group_id_colname))
    else:
        if img_class_colname is not None:
            skf = StratifiedKFold(kvs['args'].n_folds)
            splitter = skf.split(X=kvs['metadata'],
                                 y=getattr(kvs['metadata'], img_class_colname, None))
        else:
            kf = KFold(kvs['args'].n_folds)
            splitter = kf.split(X=kvs['metadata'])

    cv_split = []
    for fold_id, (train_ind, val_ind) in enumerate(splitter):

        if kvs['args'].fold != -1 and fold_id != kvs['args'].fold:
            continue

        np.random.shuffle(train_ind)
        train_ind = train_ind[::kvs['args'].skip_train]

        cv_split.append((fold_id,
                         kvs['metadata'].iloc[train_ind],
                         kvs['metadata'].iloc[val_ind]))

        kvs.update(f'losses_fold_[{fold_id}]', None, list)
        kvs.update(f'val_metrics_fold_[{fold_id}]', None, list)

    kvs.update('cv_split', cv_split)


def init_pd_meta():
    """
    Basic implementation of metadata loading. Loads the pandas data frame and stores
    it in global KVS under the `metadata` tag.

    Returns
    -------
    out : None
    """
    kvs = GlobalKVS()
    metadata = pd.read_csv(os.path.join(kvs['args'].workdir, kvs['args'].metadata))
    kvs.update('metadata', metadata)
