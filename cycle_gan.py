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


class Discriminator(nn.Module):

    def __init__(
        self, 
        input_in_channels = 3, 
        conv_dim = 64
        ):
        super(Discriminator, self).__init__()

        self.conv1 = conv(self.input_in_channels, self.conv_dim*1, 4, 2, 2, batch_norm=False)
        self.conv2 = conv(self.conv_dim*1, self.conv_dim*2, 4, 2, 2, batch_norm = True)
        self.conv3 = conv(self.conv_dim*2, self.conv_dim*4, 4, 2, 2, batch_norm = True)
        self.conv4 = conv(self.conv_dim*4, self.conv_dim*8, 4, 2, 2, batch_norm = True)
        self.conv5 = conv(self.conv_dim*8, 1, batch_norm = False)

        nn.relu = nn.ReLU()

    def forward(
        self, 
        x
        ):


        return out