import torch
import torch.nn as nn
import torch.nn.functional as F


def expand_index(mask):
    kernel = torch.ones((1, 1, 3, 3), device=mask.device).detach()
    return F.conv2d(mask.float(), weight=kernel, stride=1, padding=1)
