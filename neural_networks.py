
import numpy as np

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
        # flatten image
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

def real_loss(
    D_out, 
    smooth = False
    ):
    """ Loss for real data
        Arguments
        ---------
        D_out : outputs of Discriminator for real data
        smooth : label-smoothing

        Returns
        -------
        loss
    """
    batch_size = D_out.shape[0]
    if smooth:
        p = 0.9
    else:
        p = 1.0

    criterion = nn.BCEWithLogitsLoss()

    loss = criterion(D_out.squeeze(), torch.ones(batch_size)*p)

    return loss

def fake_loss(
    D_out
    ):
    """ Loss for fake data from Generator
        Arguments
        ---------
        D_out : outputs of Discriminator for fake data

        Returns
        -------
        loss
    """
    batch_size = D_out.shape[0]

    criterion = nn.BCEWithLogitsLoss()

    loss = criterion(D_out.squeeze(), torch.zeros(batch_size))

    return loss


def train(
    train_loader,
    D,
    G,
    n_epochs = 10,
    lr = 0.001,
    random_seed = None
    ):

    d_optimizer = optim.Adam(D.parameters(), lr = lr)
    g_optimizer = optim.Adam(G.parameters(), lr = lr)

    rng = np.random.default_rng(random_seed)

    for e in range(n_epochs):

        for inputs, _ in train_loader:
            batch_size = len(inputs)

            outputs = D(inputs)
            r_loss = real_loss(outputs)


            z = rng.uniform(0, 1, (batch_size, G.input_size))
            z = torch.from_numpy(z)
            outputs = G(z)
            f_loss = fake_loss(outputs)

            g_loss = r_loss + f_loss

            d_optimizer.zero_grad()
            g_loss.backward()
            d_optimizer.step()