import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv

class GNN(nn.Module):

    def __init__(self, dataset):
        """
        Three graph convolution layers
        34 > 4 > 4 > 2
        
        :param self: Description
        :param dataset: Input Dataset dataset = KarateClub() from torch_geometric.datasets
        """
        super().__init__()
        torch.manual_seed(1234)
        self.conv1 = GCNConv(dataset.num_features, 4)
        self.conv2 = GCNConv(4,4)
        self.conv3 = GCNConv(4,2)
        self.fc = nn.Linear(2, dataset.num_classes)

    def forward(self, x, edge_index):
        h = self.conv1(x, edge_index)
        h = h.tanh()
        h = self.conv2(h, edge_index)
        h = h.tanh()
        h = self.conv3(h, edge_index)
        h = h.tanh()
        out = self.fc(h)
        return out, h