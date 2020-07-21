import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class Discriminator(nn.Module):
    
    def __init__(
        self,
        input_size,
        hidden_dim,
        output_size
        ):
        super(Discriminator, self).__init__()
        self.input_size = input_size
        self.hidden_dim = hidden_dim
        self.output_size = output_size

        # define all layers
        self.fc1 = nn.Linear(input_size, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_size)

        self.leakyrelu = nn.LeakyReLU(0.1)

    def forward(
        self, 
        x
        ):
        # flatten imaga
        x = x.view(-1, self.input_size)
        # pass x through all layers
        x = self.fc1(x)
        x = self.leakyrelu(x)
        x = self.fc2(x)

        return x


class Generator(nn.Module):

    def __init__(
        self,
        input_size,
        hidden_dim,
        output_size
        ):
        super(Generator, self).__init__()
        self.input_size = input_size
        self.hidden_dim = hidden_dim
        self.output_size = output_size

        # define all layers
        self.fc1 = nn.Linear(input_size, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_size)

        self.leakyrelu = nn.LeakyReLU(0.1)
        self.tanh = nn.Tanh()

    def forward(
        self, 
        x
        ):
        # pass x through all layers
        x = self.fc1(x)
        x = self.leakyrelu(x)
        x = self.fc2(x)
        # final layer should have tanh applied
        x = self.tanh(x)

        return x

