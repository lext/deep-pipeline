from tqdm import tqdm
import numpy as np
import os
from termcolor import colored
from torch.utils.data import DataLoader, Dataset
import torch


def init_mean_std(snapshots_dir: str, dataset: Dataset, batch_size: int, n_threads: int, n_classes: int):
    """
    Calculates mean and std for the dataset. If the masks are available, calculates also the class weights.

    Parameters
    ----------
    snapshots_dir
    dataset
    batch_size
    n_threads
    n_classes

    Returns
    -------

    """
    if os.path.isfile(os.path.join(snapshots_dir, 'mean_std.npy')):
        return np.load(os.path.join(snapshots_dir, 'mean_std.npy'))
    else:
        tmp_loader = DataLoader(dataset, batch_size=batch_size, num_workers=n_threads)
        mean_vector = None
        std_vector = None
        num_pixels = None
        class_weights = None
        print(colored('==> ', 'green') + 'Calculating mean and std')

        if len(tmp_loader) == 0:
            raise ValueError('The data loader has no elements!')

        for batch in tqdm(tmp_loader, total=len(tmp_loader)):
            imgs = batch['img']

            if mean_vector is None:
                mean_vector = np.zeros(imgs.size(1))
                std_vector = np.zeros(imgs.size(1))

            for j in range(mean_vector.shape[0]):
                mean_vector[j] += imgs[:, j, :, :].mean()
                std_vector[j] += imgs[:, j, :, :].std()

            if 'mask' in batch:
                masks = batch['mask']
                if class_weights is None:
                    class_weights = np.zeros(n_classes)
                    num_pixels = 0

                for j in range(class_weights.shape[0]):
                    class_weights[j] += np.sum(masks.numpy() == j)
                num_pixels += np.prod(masks.size())

        mean_vector /= len(tmp_loader)
        std_vector /= len(tmp_loader)
        if class_weights is not None:
            class_weights /= num_pixels
            class_weights = 1 / class_weights
            class_weights /= class_weights.max()
            to_save = [mean_vector.astype(np.float32), std_vector.astype(np.float32), class_weights.astype(np.float32)]
        else:
            to_save = [mean_vector.astype(np.float32), std_vector.astype(np.float32)]

        np.save(os.path.join(snapshots_dir, 'mean_std.npy'), to_save)

    return to_save


def normalize_channel_wise(tensor: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    """Normalizes given tensor channel-wise

    Parameters
    ----------
    tensor: torch.Tensor
        Tensor to be normalized
    mean: torch.tensor
        Mean to be subtracted
    std: torch.Tensor
        Std to be divided by

    Returns
    -------
    result: torch.Tensor

    """
    if len(tensor.size()) != 3:
        raise ValueError

    for channel in range(tensor.size(0)):
        tensor[channel, :, :] -= mean[channel]
        tensor[channel, :, :] /= std[channel]

    return tensor
