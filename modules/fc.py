import torch.nn as nn
from torch.nn.utils.weight_norm import weight_norm


class FCNet(nn.Module):
    """ Simple class for non-linear fully connect network. """
    def __init__(self, dims, dropout=0.0):
        super(FCNet, self).__init__()

        layers = []
        for i in range(len(dims)-1):
            layers.append(weight_norm(nn.Linear(dims[i], dims[i+1]), dim=None))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
        self.main = nn.Sequential(*layers)

    def forward(self, x):
        return self.main(x)
