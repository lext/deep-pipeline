from sklearn.model_selection import KFold, GroupKFold, StratifiedKFold
from deeppipeline.kvs import GlobalKVS
import numpy as np


def init_folds(group_id_col=None, class_col=None):
    kvs = GlobalKVS()

    if group_id_col is not None:
        gkf = GroupKFold(kvs['args'].n_folds)
        splitter = gkf.split(X=kvs['metadata'],
                             y=getattr(kvs['metadata'], class_col, None),
                             groups=getattr(kvs['metadata'], group_id_col))
    else:
        if class_col is not None:
            skf = StratifiedKFold(kvs['args'].n_folds)
            splitter = skf.split(X=kvs['metadata'],
                                 y=getattr(kvs['metadata'], class_col, None))
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
