from deeppipeline.kvs import GlobalKVS
from ._losses import CombinedLoss, BCEWithLogitsLoss2d, SoftJaccardLoss, FocalLoss


def init_binary_loss():
    kvs = GlobalKVS()
    if kvs['args'].n_classes == 2:
        if kvs['args'].loss == 'combined':
            return CombinedLoss([BCEWithLogitsLoss2d(),
                                 SoftJaccardLoss(use_log=kvs['args'].log_jaccard)],
                                weights=[1-kvs['args'].loss_weight,
                                kvs['args'].loss_weight])
        elif kvs['args'].loss == 'bce':
            return BCEWithLogitsLoss2d()
        elif kvs['args'].loss == 'jaccard':
            return SoftJaccardLoss(use_log=kvs['args'].log_jaccard)
        elif kvs['args'].loss == 'focal':
            return FocalLoss()
        else:
            raise NotImplementedError
    else:
        raise NotImplementedError
