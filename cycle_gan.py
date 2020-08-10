""" #CycleGAN, Image-to-Image Translation

"""
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import DataLoader

import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms

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

        self.input_in_channels = input_in_channels
        self.conv_dim = conv_dim

        # Define all convolutional layers
        self.conv1 = conv(self.input_in_channels, self.conv_dim*1, 4, 2, 1, batch_norm = False)
        self.conv2 = conv(self.conv_dim*1, self.conv_dim*2, 4, 2, 1, batch_norm = True)
        self.conv3 = conv(self.conv_dim*2, self.conv_dim*4, 4, 2, 1, batch_norm = True)
        self.conv4 = conv(self.conv_dim*4, self.conv_dim*8, 4, 2, 1, batch_norm = True)

        # kernel_size = 128/2/2/2/2 = 8
        self.conv5 = conv(self.conv_dim*8, self.conv_dim*16, 4, 2, 1, batch_norm = False)

        self.fc_last = nn.Linear(self.conv_dim*16*4*4, 1)

        self.relu = nn.ReLU()

    def forward(
        self, 
        x
        ):

        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = self.relu(self.conv4(x))
        x = self.relu(self.conv5(x))
        out = self.fc_last(x)
        return out


class ResidualBlock(nn.Module):


    def __init__(
        self,
        conv_dim
        ):
        super(ResidualBlock, self).__init__()

    def forward(
        self
        ):
        pass



def get_data_loader(
    image_type,
    image_dir,
    image_size = 128,
    batch_size = 16,
    num_workers = 0
    ):
    """Returns training and test data loaders for a given image type. 
       These images will be resized to 128x128x3, by default, 
       converted into Tensors, and normalized.
    """
    # resize and normalize the images
    transform = transforms.Compose([transforms.Resize(image_size), # resize to 128x128
                                    transforms.ToTensor()#,
                                    # transforms.Normalize(
                                    #     mean = [0.485, 0.456, 0.406],
                                    #     std=[0.229, 0.224, 0.225]
                                    # )
                                    ])

    # get training and test directories
    image_path = os.path.join('.', image_dir)
    train_path = os.path.join(image_path, image_type)
    test_path = os.path.join(image_path, 'test_{}'.format(image_type))

    # define datasets using ImageFolder
    train_dataset = datasets.ImageFolder(train_path, transform)
    test_dataset = datasets.ImageFolder(test_path, transform)

    # create and return DataLoaders
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, test_loader


# helper scale function
def scale(
    x, 
    feature_range=(-1, 1)
    ):
    ''' Scale takes in an image x and returns that image, scaled
       with a feature_range of pixel values from -1 to 1. 
       This function assumes that the input x is already scaled from 0-255.'''
    
    # scale from 0-1 to feature_range
    min, max = feature_range
    x = x * (max - min) + min
    return x    