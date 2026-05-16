import os
import numpy as np
import torch
import math

def mse(x, y):
    return np.mean(np.abs(x - y) ** 2)

def psnr(x, y):
    assert x.shape == y.shape
    assert x.dtype == y.dtype or (np.issubdtype(x.dtype, np.float) and np.issubdtype(y.dtype, np.float))
    if x.dtype == np.uint8:
        max_intensity = 256
    else:
        max_intensity = 1
    mse = np.sum((x - y) ** 2).astype(float) / x.size
    return 20 * np.log10(max_intensity) - 10 * np.log10(mse)
import logging
from datetime import datetime

def get_timestamp():
    return datetime.now().strftime('%y%m%d-%H%M%S')

def setup_logger(logger_name, root, phase, level=logging.INFO, screen=False, tofile=False):
    lg = logging.getLogger(logger_name)
    formatter = logging.Formatter('%(asctime)s.%(msecs)03d - %(levelname)s: %(message)s', datefmt='%y-%m-%d %H:%M:%S')
    lg.setLevel(level)
    if tofile:
        log_file = os.path.join(root, phase + '_{}.log'.format(get_timestamp()))
        fh = logging.FileHandler(log_file, mode='w')
        fh.setFormatter(formatter)
        lg.addHandler(fh)
    if screen:
        sh = logging.StreamHandler()
        sh.setFormatter(formatter)
        lg.addHandler(sh)
import math
import time

def generate_measurement_matrix(size_images, subrate):
    M = round(math.sqrt(subrate * size_images * size_images))
    Phi = torch.randn(size_images, size_images)
    (q, _) = torch.qr(Phi)
    Phi = q[:M, :]
    return Phi

def complex_psnr(x, y, peak='normalized'):
    mse = np.mean(np.abs(x - y) ** 2)
    if peak == 'max':
        return 10 * np.log10(np.max(np.abs(x)) ** 2 / mse)
    else:
        return 10 * np.log10(1.0 / mse + 1e-05)

def nrmse(outputs, targets):
    outputs = outputs.reshape(-1)
    targets = targets.reshape(-1)
    if outputs.size() != targets.size():
        raise ValueError(u'Ouputs and targets tensors don have the same number of elements')
    var = torch.std(targets) ** 2
    error = (targets - outputs) ** 2
    return float(math.sqrt(torch.mean(error) / var))