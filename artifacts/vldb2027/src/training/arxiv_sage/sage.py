"""PyG SAGEConv parameterization with a weighted sparse aggregation interface."""
from common import torch
from blocks import aggregate
from torch import nn
from torch.nn import functional as F


class WeightedSAGE(nn.Module):
    # Exact default PyG mean SAGEConv: lin_l(mean)+lin_r(root), bias only lin_l.
    def __init__(self, inputs, outputs):
        super().__init__()
        from torch_geometric.nn import SAGEConv
        native = SAGEConv(inputs, outputs)
        self.lin_l, self.lin_r = native.lin_l, native.lin_r

    def forward(self, x, block):
        mean, root = aggregate(x, block)
        return self.lin_l(mean) + self.lin_r(x[root])


class SAGE(nn.Module):
    def __init__(self, inputs=128, hidden=256, outputs=40, dropout=.5):
        super().__init__()
        self.convs = nn.ModuleList([WeightedSAGE(inputs, hidden), WeightedSAGE(hidden, hidden), WeightedSAGE(hidden, outputs)])
        self.bns = nn.ModuleList([nn.BatchNorm1d(hidden), nn.BatchNorm1d(hidden)])
        self.dropout = dropout

    def forward(self, x, blocks, observer=None):
        if len(blocks) != 3:
            raise ValueError('SAGE requires three forward blocks')
        for i, (conv, block) in enumerate(zip(self.convs, blocks)):
            if observer is not None:
                observer(i, x, block)
            x = conv(x, block)
            if i < 2:
                x = F.dropout(F.relu(self.bns[i](x)), p=self.dropout, training=self.training)
        return x.log_softmax(dim=-1)
