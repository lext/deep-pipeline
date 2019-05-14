import cv2
import numpy as np


def read_gs_ocv(fname):
    return np.expand_dims(cv2.imread(fname, 0), -1)


def read_rgb_ocv(fname):
    return cv2.cvtColor(cv2.imread(fname, 0), cv2.COLOR_BGR2RGB)


def read_gs_binary_mask_ocv(fname):
    return np.expand_dims((cv2.imread(fname, 0) > 0).astype(np.float32), -1)


def read_3d_stack(slices):
    stack = None

    for i, fname in enumerate(slices):
        img = cv2.imread(fname, 0)

        if stack is None:
            stack = np.zeros((img.shape[0], len(slices), img.shape[1]), dtype=np.float16)
        stack[:, i, :] = img

    stack -= stack.min()
    stack /= stack.max()

    return stack
