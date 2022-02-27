"""
Author: Isabella Liu 1/27/22
Feature: LCN module from active stereo net module
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from .warp_ops import apply_disparity_cu
from .reprojection import apply_disparity


def local_contrast_norm(image, kernel_size=9, eps=1e-5):
    """compute local contrast normalization
    input:
        image: torch.tensor (batch_size, 1, height, width)
    output:
        normed_image
    """
    assert (kernel_size % 2 == 1), "Kernel size should be odd"
    batch_size, channel, height, width = image.shape
    if channel > 1:
        image = image[:, :1, :, :]
    batch_size, channel, height, width = image.shape
    assert (channel == 1), "Only support single channel image for now"
    unfold = nn.Unfold(kernel_size, padding=(kernel_size - 1) // 2)
    unfold_image = unfold(image)  # (batch, kernel_size*kernel_size, height*width)
    avg = torch.mean(unfold_image, dim=1).contiguous().view(batch_size, 1, height, width)
    std = torch.std(unfold_image, dim=1, unbiased=False).contiguous().view(batch_size, 1, height, width)

    normed_image = (image - avg) / (std + eps)
    return normed_image, std