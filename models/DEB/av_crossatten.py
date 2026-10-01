import torch.nn as nn
import math
import torch
import sys


class DenseCoAttn(nn.Module):

    def __init__(self, dim1, dim2, dropout):
        super(DenseCoAttn, self).__init__()
        dim = dim1 + dim2
        self.dropouts = nn.ModuleList([nn.Dropout(p=dropout) for _ in range(2)])
        self.query_linear = nn.Linear(dim, dim)

        self.key1_linear = nn.Linear(dim1, dim1)
        self.key2_linear = nn.Linear(dim2, dim2)

        self.value1_linear = nn.Linear(dim1, dim1)
        self.value2_linear = nn.Linear(dim2, dim2)

        self.relu = nn.ReLU()

    def forward(self, value1, value2):

        joint = torch.cat((value1, value2), dim=-1)

        # audio  audio*W*joint
        va_joint = self.query_linear(joint)

        key1 = self.key1_linear(value1).transpose(1, 2)
        key2 = self.key2_linear(value2).transpose(1, 2)

        value1 = self.value1_linear(value1)
        value2 = self.value2_linear(value2)

        weighted1, attn1 = self.qkv_attention(joint, key1, value1, dropout=self.dropouts[0])
        weighted2, attn2 = self.qkv_attention(joint, key2, value2, dropout=self.dropouts[1])

        return weighted1, weighted2

    def qkv_attention(self, query, key, value, dropout=None):
        d_k = query.size(-1)
        scores = torch.bmm(key, query) / math.sqrt(d_k)
        scores = torch.tanh(scores)
        if dropout:
            scores = dropout(scores)

        weighted = torch.tanh(torch.bmm(value, scores))
        return self.relu(weighted), scores


class NormalSubLayer(nn.Module):
    def __init__(self, dim1, dim2, dropout):
        super(NormalSubLayer, self).__init__()
        self.dense_coattn = DenseCoAttn(dim1, dim2, dropout)
        self.linears = nn.ModuleList([
            nn.Sequential(
                nn.Linear(dim1 + dim2, dim1),
                nn.ReLU(inplace=True),
                nn.Dropout(p=dropout),
            ),
            nn.Sequential(
                nn.Linear(dim1 + dim2, dim2),
                nn.ReLU(inplace=True),
                nn.Dropout(p=dropout),
            )
        ])

    def forward(self, data1, data2):
        weighted1, weighted2 = self.dense_coattn(data1, data2)
        data1 = data1 + self.linears[0](weighted1)
        data2 = data2 + self.linears[1](weighted2)

        return data1, data2


class DCNLayer(nn.Module):

    def __init__(self, dim1, dim2, num_seq, dropout):
        super(DCNLayer, self).__init__()
        self.dcn_layers = nn.ModuleList([NormalSubLayer(dim1, dim2, dropout) for _ in range(num_seq)])

    def forward(self, data1, data2):
        for dense_coattn in self.dcn_layers:
            data1, data2 = dense_coattn(data1, data2)

        return data1, data2
