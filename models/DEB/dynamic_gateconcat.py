import torch
from torch import nn
from torch.nn import init
from torch.nn import functional as F


class DynamicWeightedConcat2(nn.Module):
    def __init__(self, feat1_dim, feat2_dim, hidden_dim=64, weight_strength=0.1):
        super().__init__()
        self.weight_strength = weight_strength
        combined_dim = feat1_dim + feat2_dim
        self.weight_net = nn.Sequential(
            nn.Linear(combined_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2)
        )

    def forward(self, x1, x2):
        combined = torch.cat([x1, x2], dim=-1)
        scores = self.weight_net(combined)
        weights = torch.softmax(scores, dim=-1)
        weight1 = weights[..., 0].unsqueeze(-1)  # (batch_size, seq_len, 1)
        weight2 = weights[..., 1].unsqueeze(-1)  # (batch_size, seq_len, 1)

        weighted_x1 = x1 + self.weight_strength * (x1 * weight1)
        weighted_x2 = x2 + self.weight_strength * (x2 * weight2)

        output = torch.cat([weighted_x1, weighted_x2], dim=-1)

        return output
