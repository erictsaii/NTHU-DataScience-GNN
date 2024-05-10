import torch
import torch.nn as nn
import torch.nn.functional as F

from dgl.nn.pytorch import GraphConv
from torch_geometric.nn import GCNConv


class GCN(nn.Module):
    """
    Baseline Model:
    - A simple two-layer GCN model, similar to https://github.com/tkipf/pygcn
    - Implement with DGL package
    """

    def __init__(self, in_size, hid_size, out_size):
        super().__init__()
        self.layers = nn.ModuleList()
        # two-layer GCN
        self.layers.append(GraphConv(in_size, hid_size, activation=F.relu))
        self.layers.append(GraphConv(hid_size, out_size))
        self.dropout = nn.Dropout(0.5)

    def forward(self, g, features):
        h = features
        for i, layer in enumerate(self.layers):
            if i != 0:
                h = self.dropout(h)
            h = layer(g, h)
        return h


# -----------------------------------------------------------------------------------------


class CRD(torch.nn.Module):

    def __init__(self, d_in, d_out, p):
        super(CRD, self).__init__()
        self.conv = GCNConv(in_channels=d_in, out_channels=d_out, cached=True)
        self.p = p

    def forward(self, x, edge_index):
        x = F.relu(self.conv(x, edge_index))
        x = F.dropout(x, p=self.p, training=self.training)
        return x


class CLS(torch.nn.Module):

    def __init__(self, d_in, d_out):
        super(CLS, self).__init__()
        self.conv = GCNConv(in_channels=d_in, out_channels=d_out, cached=True)

    def forward(self, x, edge_index):
        x = self.conv(x, edge_index)
        x = F.log_softmax(x, dim=1)
        return x


class Net(torch.nn.Module):

    def __init__(self, din, dhid, dout):
        super(Net, self).__init__()
        self.crd = CRD(din, dhid, 0.5)
        self.cls = CLS(dhid, dout)

    def forward(self, x, edge_index):
        x = self.crd(x, edge_index)
        x = self.cls(x, edge_index)
        return x
