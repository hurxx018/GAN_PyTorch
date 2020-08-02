import torch
import torch.nn as nn
import torch.optim as optim


class Discriminator(nn.Module):

    def __init__(self):
        super().__init__(
            hidden_dim = 32
        )
        self.hidden_dim = hidden_dim
        self.conv1 = nn.Conv2d(3, self.hidden_dim, 2, 2)
        self.conv2 = nn.Conv2d(self.hidden_dim, self.hidden_dim*2, 2, 2)
        self.conv3 = nn.Conv2d(self.hidden_dim*2, self.hidden_dim*4, 2, 2)

    def forward(self, x):


        return


class Generator(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x):


        return