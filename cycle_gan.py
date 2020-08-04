""" #CycleGAN, Image-to-Image Translation

"""
import torch.nn as nn
import torch.nn.functional as F


def conv(
    in_channels,
    out_channels,
    kernel_size,
    stride = 2,
    padding = 1,
    batch_norm = True
    ):
    """ Create a convolutional layer with an optional batch normalization layer.
    """
    layers = []
    conv_layer = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
    layers.append(conv_layer)

    if batch_norm:
        layers.append(nn.BatchNorm2d(out_channels))
    return nn.Sequential(*layers)

